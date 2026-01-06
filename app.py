import os
import sys
import requests
import io  # Memory buffer

# Spaces environment configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import random
import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_url, login  # hf_hub_url for generating cloud URL
import gradio as gr

# Attempt to import Anomagic (if ip_adapter module exists)
try:
    from ip_adapter.ip_adapter_anomagic import Anomagic

    HAS_ANOMAGIC = True
except ImportError:
    HAS_ANOMAGIC = False
    print("Anomagic not imported, will use basic Inpainting")

# Get the absolute path of the current script (to solve path issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_image_from_editor_output(editor_output):
    """Extract PIL Image from gr.ImageEditor output (can be dict or PIL Image)"""
    if editor_output is None:
        return None

    # 如果已经是 PIL Image，直接返回
    if isinstance(editor_output, Image.Image):
        return editor_output

    # 如果是字典（gr.ImageEditor 的输出格式）
    if isinstance(editor_output, dict):
        # gr.ImageEditor 返回格式：{"background": image, "layers": [], "composite": image}
        # 优先使用 composite（合成后的图像）
        if "composite" in editor_output and editor_output["composite"] is not None:
            return editor_output["composite"]
        elif "background" in editor_output and editor_output["background"] is not None:
            return editor_output["background"]

    # 如果是其他格式但可转换为图像
    try:
        return Image.fromarray(editor_output)
    except:
        pass

    return None


class SingleAnomalyGenerator:
    def __init__(self, device="cuda:0"):
        # Auto-detect GPU and set dtype
        if torch.cuda.is_available() and "cuda" in device:
            self.device = torch.device(device)
            self.dtype = torch.float16
            print(f"Using GPU: {device}, dtype: {self.dtype}")
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            print(f"Using CPU, dtype: {self.dtype}")

        self.anomagic_model = None
        self.pipe = None  # Save pipe for reuse
        self.clip_vision_model = None
        self.clip_image_processor = None
        self.ip_ckpt_path = None  # IP weights state_dict in memory
        self.att_ckpt_path = None  # ATT weights state_dict in memory

    def load_models(self):
        """Load models with official CLIP"""
        print("Loading VAE...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=self.dtype
        ).to(self.device)

        print("Loading base model...")
        from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, DPMSolverMultistepScheduler

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            "SG161222/Realistic_Vision_V4.0_noVAE",
            torch_dtype=self.dtype,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            low_cpu_mem_usage=True
        ).to(self.device, dtype=self.dtype)

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        print("Loading CLIP image encoder...")
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=self.dtype
        ).to(self.device)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        print("All models loaded!")

        # Load weights (download from cloud repo to memory, avoid any disk usage)
        print("Loading weights into memory...")
        weight_files = [
            ("checkpoint/anomagic.bin", "ip_ckpt_path"),
            ("checkpoint/attention_module.bin", "att_ckpt_path")
        ]
        for filename, attr_name in weight_files:
            try:
                # Generate cloud URL (public repo, no token needed)
                repo_id = "yuxinjiang11/Anomagic_model"
                url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="model")

                # Dynamically set attribute (or use if to assign explicitly)
                if attr_name == "ip_ckpt_path":
                    self.ip_ckpt_path = url
                elif attr_name == "att_ckpt_path":
                    self.att_ckpt_path = url

                print(f"Weight file path: {filename} -> {url}")
            except Exception as e:
                raise FileNotFoundError(f"Unable to get weight file path {filename}: {str(e)}")

        # If Anomagic is available, load weights into the model
        if HAS_ANOMAGIC:
            print("Initializing Anomagic model...")
            self.anomagic_model = Anomagic(self.pipe, self.clip_vision_model, self.ip_ckpt_path, self.att_ckpt_path,
                                           self.device)
        else:
            print("No Anomagic, using basic Pipe.")

        print("Model loading complete!")

    def generate_single_image(self, normal_image, reference_image, mask, mask_0, prompt, num_inference_steps=50,
                              ip_scale=0.3, seed=42, strength=0.3):
        """Generate anomaly image with mask_0 support for reference image mask."""
        if normal_image is None or reference_image is None:
            raise ValueError("Normal or reference image is None. Please upload valid images.")

        target_size = (512, 512)
        normal_image = normal_image.resize(target_size)
        reference_image = reference_image.resize(target_size)

        # Process normal image mask
        if mask is not None:
            mask = extract_image_from_editor_output(mask)
            if mask is not None and isinstance(mask, Image.Image):
                mask = mask.resize(target_size)
                mask = mask.convert('L')
                mask = np.array(mask) > 0
                mask = Image.fromarray(mask.astype(np.uint8) * 255).convert('L')

        # Process reference image mask (mask_0)
        if mask_0 is not None:
            mask_0 = extract_image_from_editor_output(mask_0)
            if mask_0 is not None and isinstance(mask_0, Image.Image):
                mask_0 = mask_0.resize(target_size)
                mask_0 = mask_0.convert('L')
                mask_0 = np.array(mask_0) > 0
                mask_0 = Image.fromarray(mask_0.astype(np.uint8) * 255).convert('L')

        print(f"Generating with seed {seed}...")
        torch.manual_seed(seed)

        # If Anomagic is available, use it to generate; otherwise basic Inpainting
        if HAS_ANOMAGIC and self.anomagic_model:
            # generator = torch.Generator(device=self.device).manual_seed(seed)
            # Assume Anomagic.generate supports parameters (adjust based on actual)
            generated_image = self.anomagic_model.generate(
                pil_image=reference_image,
                num_samples=1,
                num_inference_steps=num_inference_steps,
                prompt=prompt,
                scale=ip_scale,
                image=normal_image,
                mask_image=mask,
                mask_image_0=mask_0,  # Reference image mask
                strength=strength,
                # generator=generator
            )[0]
        else:
            # Basic Inpainting
            # generator = torch.Generator(device=self.device).manual_seed(seed)
            if mask is None:
                mask = Image.new('L', target_size, 255)  # Full white mask
            generated_image = self.pipe(
                prompt=prompt,
                image=normal_image,
                mask_image=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                # generator=generator,
            ).images[0]

        return generated_image


# Global generator and load status
generator = None
load_status = {"loaded": False, "error": None}


def load_generator():
    """Background load function: Automatically load model on startup"""
    global generator, load_status

    if load_status["loaded"]:
        return "Models loaded!"

    if load_status["error"]:
        return f"Previous load failed: {load_status['error']}"

    try:
        print("Starting background model load...")
        generator = SingleAnomalyGenerator()
        generator.load_models()
        load_status["loaded"] = True
        print("Background model load complete!")
        return "Model loading complete! You can now generate images."
    except Exception as e:
        load_status["error"] = str(e)
        error_msg = f"Model loading failed: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        return error_msg


def generate_random_mask(size=(512, 512), num_blobs=3, blob_size_range=(50, 150)):
    """Generate random mask: Create several random blobs as anomaly areas"""
    mask = Image.new('L', size, 0)  # Black background
    draw = ImageDraw.Draw(mask)
    for _ in range(num_blobs):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        width = random.randint(*blob_size_range)
        height = random.randint(*blob_size_range)
        # Draw elliptical blobs
        draw.ellipse([x - width // 2, y - height // 2, x + width // 2, y + height // 2], fill=255)
    return mask


def generate_anomaly(normal_img, reference_img, mask_img, mask_0_img, prompt, strength, ip_scale, steps, seed):
    """Core generation function: Called by Gradio (supports two masks)"""
    global generator

    if not load_status["loaded"]:
        return None, "Please wait for model loading to complete."

    if normal_img is None or reference_img is None or not prompt.strip():
        return None, "Please upload normal image, reference image, and enter prompt text."

    if mask_img is None:
        return None, "Please upload or generate mask image for normal image."

    try:
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        generated_img = generator.generate_single_image(
            normal_image=normal_img,
            reference_image=reference_img,
            mask=mask_img,
            mask_0=mask_0_img,
            prompt=prompt,
            num_inference_steps=steps,
            ip_scale=ip_scale,
            seed=seed,
            strength=strength
        )

        return generated_img, f"Generation successful! Seed: {seed}, Steps: {steps}"

    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        return None, error_msg


# Predefined anomaly examples (using local image paths; assume images are in examples/ folder in the same directory as the script)
EXAMPLE_PAIRS = [
    {
        "normal": "examples/normal_apple.png",  # Your local normal gear image
        "reference": "examples/reference_apple.png",  # Your local rusty gear reference image
        "mask": "examples/normal_mask_apple.jpg",  # Your local mask for normal gear
        "mask_0": "examples/ref_mask_apple.png",  # Your local mask for reference gear
        "prompt": "Wood surface has holes with rough - edged circular openings.",
        "strength": 0.6,
        "ip_scale": 0.1,
        "steps": 20,
        "seed": 42,
        "description": "Apple with wormholes and rough edges"
    },
    {
        "normal": "examples/normal_candle.JPG",  # Your local normal gear image
        "reference": "examples/reference_candle.png",  # Your local rusty gear reference image
        "mask": "examples/normal_mask_candle.png",  # Your local mask for normal gear
        "mask_0": "examples/ref_mask_candle.png",  # Your local mask for reference gear
        "prompt": "Chocolate - chip cookie has a chunk - missing defect with exposed inner texture. ",
        "strength": 0.6,
        "ip_scale": 1,
        "steps": 20,
        "seed": 42,
        "description": "Candle with deformed surface"
    },
    {
        "normal": "examples/normal_wood.png",  # Your local normal gear image
        "reference": "examples/reference_wood.png",  # Your local rusty gear reference image
        "mask": "examples/normal_mask_wood.png",  # Your local mask for normal gear
        "mask_0": "examples/ref_mask_wood.png",  # Your local mask for reference gear
        "prompt": "Wood surface has a crack with a long, dark - hued split.",
        "strength": 0.6,
        "ip_scale": 0.1,
        "steps": 20,
        "seed": 42,
        "description": "Wood with long dark crack and split"
    },
]


def load_example(idx):
    """Load example: Load images from local path, generate random mask if not provided, and set UI"""
    if idx >= len(EXAMPLE_PAIRS):
        return None, None, None, None, "", 0.5, 0.3, 20, 42, f"Example {idx + 1} not found"

    ex = EXAMPLE_PAIRS[idx]
    try:
        # Load normal image
        normal_img = Image.open(ex["normal"]).convert('RGB')

        # Load reference image
        reference_img = Image.open(ex["reference"]).convert('RGB')

        # Load or generate normal mask
        if ex["mask"] is not None:
            mask_img = Image.open(ex["mask"]).convert('L')
        else:
            mask_img = generate_random_mask()

        # Load or generate reference mask (mask_0)
        if ex["mask_0"] is not None:
            mask_0_img = Image.open(ex["mask_0"]).convert('L')
        else:
            mask_0_img = generate_random_mask()

        return normal_img, reference_img, mask_img, mask_0_img, ex["prompt"], ex["strength"], ex["ip_scale"], ex[
            "steps"], ex["seed"], f"Example {idx + 1}: {ex['description']} loaded!"
    except Exception as e:
        error_msg = f"Example loading failed: {str(e)} (Check if local image paths are correct)"
        print(error_msg)
        # Fallback to placeholder images and random masks
        normal_img = Image.new('RGB', (512, 512), color='gray')
        reference_img = Image.new('RGB', (512, 512), color='blue')
        mask_img = generate_random_mask()
        mask_0_img = generate_random_mask()
        return normal_img, reference_img, mask_img, mask_0_img, ex["prompt"], ex["strength"], ex["ip_scale"], ex[
            "steps"], ex["seed"], error_msg


# Automatically load model on startup
load_generator()

# Gradio UI
with gr.Blocks(title="Anomagic Anomaly Image Generator") as demo:  # Removed theme to fix compatibility
    gr.Markdown("# Anomagic: Single Anomaly Image Generation Demo")
    gr.Markdown(
        "Upload normal image, reference image, normal mask and reference mask (white areas are for inpainting/anomaly generation), enter prompt, adjust parameters, and generate synthetic anomaly images with one click. Model is loaded in the background.")

    with gr.Row():
        with gr.Column(scale=1):
            normal_img = gr.Image(type="pil", label="Normal Image", height=256)  # Limit height
            reference_img = gr.Image(type="pil", label="Reference Image", height=256)

            with gr.Row():  # Mask row: Add buttons
                mask_img = gr.ImageEditor(
                    type="pil",
                    sources=['upload', 'webcam', 'clipboard'],
                    label="Normal Image Mask (draw white anomaly areas on black background)",
                    height=256,
                    interactive=True,
                    brush=gr.Brush(default_color="white", default_size=15, color_mode="fixed"),
                    value=Image.new('L', (512, 512), 0)  # Initial black canvas
                )
                with gr.Row():
                    generate_mask_btn = gr.Button("Generate Random Normal Mask", variant="secondary")
                    clear_mask_btn = gr.Button("Clear Normal Mask", variant="secondary")

            mask_0_img = gr.ImageEditor(
                type="pil",
                sources=['upload', 'webcam', 'clipboard'],
                label="Reference Image Mask (draw white areas on black background)",
                height=256,
                interactive=True,
                brush=gr.Brush(default_color="white", default_size=15, color_mode="fixed"),
                value=Image.new('L', (512, 512), 0)  # Initial black canvas
            )
            with gr.Row():
                generate_mask_0_btn = gr.Button("Generate Random Reference Mask", variant="secondary")
                clear_mask_0_btn = gr.Button("Clear Reference Mask", variant="secondary")

            prompt = gr.Textbox(label="Prompt Text",
                                placeholder="e.g., a broken machine part with rust and cracks")

        with gr.Column(scale=1):
            strength = gr.Slider(0.1, 1.0, value=0.5, label="Denoising Strength")
            ip_scale = gr.Slider(0, 2.0, value=0.3, step=0.1, label="IP Adapter Scale")
            steps = gr.Slider(10, 100, value=20, step=5, label="Inference Steps")
            seed = gr.Slider(0, 2 ** 32 - 1, value=42, step=1, label="Random Seed")

            gr.Markdown("## Examples")
            gr.Markdown(
                "Click the buttons below to load predefined examples for quick testing. After loading, click 'Generate Image' to view the anomaly synthesis result.")

            # Create example buttons
            example_buttons = []
            for i in range(len(EXAMPLE_PAIRS)):
                example_btn = gr.Button(f"Example {i + 1}: {EXAMPLE_PAIRS[i]['description']}", variant="secondary")
                example_buttons.append(example_btn)

            # 定义输出组件
            output_img = gr.Image(type="pil", label="Generated Anomaly Image", height=256)
            status = gr.Textbox(label="Status", interactive=False)

            generate_btn = gr.Button("Generate Image", variant="primary", size="lg")  # Enlarge button


            # Clear cache button
            def clear_cache():
                global load_status
                load_status = {"loaded": False, "error": None}
                return "Cache cleared, please restart the app to reload the model."


            clear_btn = gr.Button("Clear Cache", variant="stop")

    # 连接所有事件处理函数（在组件定义之后）

    # 生成按钮点击事件
    generate_btn.click(
        generate_anomaly,
        inputs=[normal_img, reference_img, mask_img, mask_0_img, prompt, strength, ip_scale, steps, seed],
        outputs=[output_img, status]
    )

    # 清除缓存按钮点击事件
    clear_btn.click(clear_cache, outputs=status)

    # 示例按钮点击事件（为每个按钮单独连接）
    for i, btn in enumerate(example_buttons):
        btn.click(lambda idx=i: load_example(idx),
                  outputs=[normal_img, reference_img, mask_img, mask_0_img, prompt, strength, ip_scale, steps, seed,
                           status])

    # 掩码按钮点击事件
    generate_mask_btn.click(lambda: generate_random_mask(), outputs=mask_img)
    clear_mask_btn.click(lambda: Image.new('L', (512, 512), 0), outputs=mask_img)
    generate_mask_0_btn.click(lambda: generate_random_mask(), outputs=mask_0_img)
    clear_mask_0_btn.click(lambda: Image.new('L', (512, 512), 0), outputs=mask_0_img)

if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(server_name="0.0.0.0", server_port=7860)