import os
from typing import List
from peft import LoraConfig, LoraModel
import torch
from typing import List, Optional, Union
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn as nn
import math
from .utils import is_torch2_available, get_generator
import numpy as np
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel
if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


def load_lora_model(unet, device, diffusion_model_learning_rate):
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=diffusion_model_learning_rate,
    )
    return unet, lora_layers


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        b = embeds.shape[0]
        # clip_extra_context_tokens = self.proj(embeds).reshape(
        #     -1, self.clip_extra_context_tokens, self.cross_attention_dim
        # )
        clip_extra_context_tokens = self.proj(embeds).reshape(
            b, -1, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class SelfAttention(nn.Module):
    def __init__(self, in_channels, device):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1).to(device)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1).to(device)
        self.proj_out = nn.Linear(1280, 1024).to(device)

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        batch_size, channels, h = x.size()
        height = int(math.sqrt(h))
        width = height
        x = x.view(batch_size, channels, width, height)
        batch_size, channels, height, width = x.size()
        # 计算 query, key, value
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        # 计算注意力分数
        attention_scores = torch.bmm(q, k)

        if mask is not None:
            # 将 mask 的尺寸调整为和 x 一致
            mask = nn.functional.interpolate(mask, size=(height, width), mode='nearest')
            mask = mask.view(batch_size, 1, height * width)
            # c
            large_constant = 1e6
            attention_scores = attention_scores - (1 - mask) * large_constant

        # 计算注意力权重
        attention_weights = self.softmax(attention_scores)

        # 应用注意力权重
        out = torch.bmm(v, attention_weights.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # 加权求和
        out = self.gamma.to(x.device) * out + x
        out = out.view(batch_size, channels, height * width)
        out = out.permute(0, 2, 1)
        out = self.proj_out(out)

        return out


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, ip_ckpt_1, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.ip_ckpt_1 = ip_ckpt_1
        self.num_tokens = num_tokens
        self.attention_module = SelfAttention(1280, device)
        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )

        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)

        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        unet, lora_layers = load_lora_model(unet, self.device, 4e-4)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj_model."):
                        state_dict["image_proj"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter_model."):
                        state_dict["ip_adapter"][key.replace("ip_adapter_model.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        print("unet的参数列表：")
        for name, param in self.pipe.unet.named_parameters():
            print(name)
        self.pipe.unet.load_state_dict(state_dict["unet"])
        state_dict_1 = torch.load(self.ip_ckpt_1, map_location="cpu")
        # 打印attention_module的所有参数名称
        print("self.attention_module的参数列表：")
        for name, param in self.attention_module.named_parameters():
            print(name)
        self.attention_module.load_state_dict(state_dict_1["att"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, mask_image_0=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
            outputs = self.image_encoder(clip_image.to(self.device, dtype=torch.float16))
            clip_image_embeds = outputs.image_embeds
            last_feature_layer_output = outputs.last_hidden_state
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        mask_image_0 = mask_image_0.resize((64, 64))
        mask_image_0 = mask_image_0.convert('L')
        mask_image_0 = torch.tensor(np.array(mask_image_0), dtype=torch.float32)
        mask_image_0 = (mask_image_0 > 0.5).float().to(self.device)
        image_embeds = self.attention_module(last_feature_layer_output[:, :256, :].float(),
                                             mask_image_0.unsqueeze(0).unsqueeze(0))
        image_prompt_embeds = self.image_proj_model(image_embeds.half())
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds).half())
        # uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds).half())
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def encode_long_text(self,
            input_ids: torch.Tensor,  # 直接传入token IDs
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
            max_length: int = 77,  # CLIP的token限制
            device: str = "cuda"
    ) -> torch.Tensor:
        """
        分段编码已经tokenize的长文本，合并所有段的特征取平均

        Args:
            input_ids: 已经tokenize的输入ID [batch_size, seq_len] 或 [seq_len]
            tokenizer: CLIP的tokenizer (用于提供特殊token信息)
            text_encoder: CLIP的text_encoder
            max_length: 单段最大token数（默认77）
            device: 计算设备

        Returns:
            combined_embeddings: 合并后的文本特征 [batch_size, hidden_dim]
        """
        # 确保输入是2D tensor [batch_size, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]

        batch_size = input_ids.size(0)
        hidden_dim = text_encoder.config.hidden_size

        # 初始化结果张量
        combined_embeddings = torch.zeros(batch_size, hidden_dim, device=device)

        for batch_idx in range(batch_size):
            # 获取当前batch的token IDs
            current_input_ids = input_ids[batch_idx]  # [seq_len]

            # 1. 按max_length分段
            chunks = [
                current_input_ids[i:i + max_length]
                for i in range(0, len(current_input_ids), max_length)
            ]

            # 2. 对每段编码并收集特征
            embeddings = []
            for chunk in chunks:
                # 添加batch维度并padding到max_length
                chunk_len = len(chunk)
                padding_len = max_length - chunk_len

                # 构建模型输入
                chunk_input = {
                    "input_ids": torch.cat([
                        chunk.unsqueeze(0).to(device),  # [1, chunk_len]
                        torch.zeros(1, padding_len, dtype=torch.long, device=device)  # [1, padding_len]
                    ], dim=1),  # [1, max_length]

                    "attention_mask": torch.cat([
                        torch.ones(1, chunk_len, dtype=torch.long, device=device),  # [1, chunk_len]
                        torch.zeros(1, padding_len, dtype=torch.long, device=device)  # [1, padding_len]
                    ], dim=1)  # [1, max_length]
                }

                with torch.no_grad():
                    chunk_emb = text_encoder(**chunk_input).last_hidden_state  # [1, max_length, hidden_dim]
                    # 只取非padding部分的特征并平均
                    embeddings.append(chunk_emb[:, :chunk_len, :].mean(dim=1))

            # 3. 合并所有段的特征（平均）
            if embeddings:  # 确保有embedding
                combined_embeddings[batch_idx] = torch.mean(torch.cat(embeddings, dim=0), dim=0)
            else:
                # 处理空输入情况
                combined_embeddings[batch_idx] = torch.zeros(hidden_dim, device=device)

        return combined_embeddings.unsqueeze(1)
    # def encode_long_text(
    #         self,
    #         text: Union[str, List[str]],
    #         tokenizer: CLIPTokenizer,
    #         text_encoder: CLIPTextModel,
    #         max_length: int = 77,
    #         device: Optional[str] = None
    # ) -> torch.Tensor:
    #     """
    #     Encode long text by splitting into chunks and averaging embeddings
    #
    #     Args:
    #         text: Input text or list of texts
    #         tokenizer: CLIP tokenizer
    #         text_encoder: CLIP text encoder
    #         max_length: Maximum token length per chunk
    #         device: Device to use (defaults to self.device)
    #
    #     Returns:
    #         torch.Tensor: Text embeddings [batch_size, seq_len, hidden_dim]
    #     """
    #     device = device or self.device
    #
    #     # Tokenize input text
    #     if isinstance(text, str):
    #         text = [text]
    #
    #     # Tokenize without truncation
    #     inputs = tokenizer(
    #         text,
    #         padding=False,
    #         truncation=False,
    #         return_tensors="pt",
    #         max_length=None
    #     )
    #     input_ids = inputs.input_ids.to(device)
    #     attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
    #
    #     batch_size, seq_len = input_ids.shape
    #
    #     # Calculate number of chunks needed
    #     num_chunks = (seq_len + max_length - 1) // max_length
    #
    #     # Initialize embeddings tensor
    #     embeddings = []
    #
    #     for i in range(num_chunks):
    #         start_idx = i * max_length
    #         end_idx = (i + 1) * max_length
    #
    #         # Get chunk
    #         chunk_input_ids = input_ids[:, start_idx:end_idx]
    #         chunk_attention_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
    #
    #         # Pad if needed
    #         padding_len = max_length - chunk_input_ids.shape[1]
    #         if padding_len > 0:
    #             padding = torch.zeros(batch_size, padding_len, dtype=torch.long, device=device)
    #             chunk_input_ids = torch.cat([chunk_input_ids, padding], dim=1)
    #             if chunk_attention_mask is not None:
    #                 chunk_attention_mask = torch.cat([
    #                     chunk_attention_mask,
    #                     torch.zeros(batch_size, padding_len, dtype=torch.long, device=device)
    #                 ], dim=1)
    #
    #         # Encode chunk
    #         with torch.no_grad():
    #             outputs = text_encoder(
    #                 input_ids=chunk_input_ids,
    #                 attention_mask=chunk_attention_mask,
    #                 return_dict=True
    #             )
    #             chunk_embeddings = outputs.last_hidden_state
    #
    #             # Apply attention mask if available
    #             if chunk_attention_mask is not None:
    #                 chunk_embeddings = chunk_embeddings * chunk_attention_mask.unsqueeze(-1)
    #
    #             # Average over sequence length (excluding padding)
    #             if chunk_attention_mask is not None:
    #                 valid_lengths = chunk_attention_mask.sum(dim=1, keepdim=True)
    #                 chunk_embeddings = (chunk_embeddings.sum(dim=1) / valid_lengths.clamp(min=1))
    #             else:
    #                 chunk_embeddings = chunk_embeddings.mean(dim=1)
    #
    #             embeddings.append(chunk_embeddings)
    #
    #     # Combine chunk embeddings by averaging
    #     if embeddings:
    #         combined_embeddings = torch.stack(embeddings, dim=1)  # [batch_size, num_chunks, hidden_dim]
    #         combined_embeddings = combined_embeddings.mean(dim=1)  # [batch_size, hidden_dim]
    #     else:
    #         combined_embeddings = torch.zeros(batch_size, text_encoder.config.hidden_size, device=device)
    #
    #     return combined_embeddings.unsqueeze(1)  # Add sequence dimension [batch_size, 1, hidden_dim]

    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            # guidance_scale=10,
            num_inference_steps=30,
            mask_image_0=None,
            **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        # print("prompt:", prompt)
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds, mask_image_0=mask_image_0,
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            # 编码文本提示（统一使用长文本分段编码）
            prompt_embeds_list = []
            for p in prompt:
                # 1. 文本转 token IDs
                inputs = self.pipe.tokenizer(
                    p,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(self.device)  # [1, seq_len]

                # 2. 调用长文本编码函数（无论是否超长）
                prompt_embed = self.encode_long_text(
                    input_ids=input_ids,
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    device=self.device
                )  # 返回 [1, hidden_dim]

                prompt_embeds_list.append(prompt_embed)

            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)

            # 编码负向提示（同理，统一使用长文本处理）
            negative_prompt_embeds_list = []
            for p in negative_prompt:
                inputs = self.pipe.tokenizer(
                    p,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(self.device)

                negative_prompt_embed = self.encode_long_text(
                    input_ids=input_ids,
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    device=self.device
                )

                negative_prompt_embeds_list.append(negative_prompt_embed)

            negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=0)

            # 合并图像嵌入与文本嵌入（保持不变）
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            # prompt_embeds = torch.cat([prompt_embeds, uncond_image_prompt_embeds], dim=1)
            # prompt_embeds = prompt_embeds
            # prompt_embeds = prompt_embeds.repeat(1, 1024, 1)
            # negative_prompt_embeds = image_prompt_embeds
            # negative_prompt_embeds = self.pipe.encode_prompt(
            #     negative_prompt,
            #     device=self.device,
            #     num_images_per_prompt=num_samples,
            #     do_classifier_free_guidance=True
            # )[0]

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
