import click
import torch
from PIL import Image
import orjson
import numpy as np
from pathlib import Path
# Attempt to import Anomagic (if ip_adapter module exists)
try:
    from ip_adapter.ip_adapter_anomagic import Anomagic
    HAS_ANOMAGIC = True
except ImportError as e:
    HAS_ANOMAGIC = False
    print("Anomagic not imported, will use basic Inpainting")
    
class SingleAnomalyGenerator:
    def __init__(self, device="cuda:0"):
        # Auto-detect GPU and set dtype
        if torch.cuda.is_available() and "cuda" in device:
            self.device = torch.device(device)
            self.dtype = torch.float32
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
        self.ip_ckpt_path="checkpoints/anomagic.bin"
        self.att_ckpt_path="checkpoints/attention_module.bin"
        # If Anomagic is available, load weights into the model
        if HAS_ANOMAGIC:
            print("Initializing Anomagic model...")
            self.anomagic_model = Anomagic(self.pipe, self.clip_vision_model, self.ip_ckpt_path, self.att_ckpt_path,
                                           self.device,dtype=self.dtype)
        else:
            print("No Anomagic, using basic Pipe.")

        print("Model loading complete!")

    def generate_single_image(self, normal_image, reference_image, mask, mask_0, prompt, num_inference_steps=50,
                              ip_scale=0.3, seed=42, strength=0.3):
        """Generate anomaly image with mask_0 support for reference image mask."""
        if normal_image is None or reference_image is None:
            raise ValueError("Normal or reference image is None. Please upload valid images.")

        target_size = (2048, 2048)
        normal_image = normal_image.resize(target_size)
        reference_image = reference_image.resize(target_size)
        
        if mask is not None and isinstance(mask, Image.Image):
            mask = mask.resize(target_size)
            mask = mask.convert('L')
            mask = np.array(mask) > 100
            mask = Image.fromarray(mask.astype(np.uint8) * 255).convert('L')

        # Process reference image mask (mask_0)
        if mask_0 is not None and isinstance(mask_0, Image.Image):
            mask_0 = mask_0.resize(target_size)
            mask_0 = mask_0.convert('L')
            mask_0 = np.array(mask_0) > 100
            mask_0 = Image.fromarray(mask_0.astype(np.uint8) * 255).convert('L')

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


@click.command()
@click.option("--config_path", type=str, help="Normal image path",default="/tmp/code/screw_pictures/loc1/merge.json")
def generate_anomaly(config_path):
    generator=SingleAnomalyGenerator()
    generator.load_models()
    with open(config_path,"rb") as f:
        config=orjson.loads(f.read())
    config_dir=Path(config_path).parent
    output_dir=config_dir.joinpath("merge")
    output_dir.mkdir(exist_ok=True,parents=True)
    prompt=config["prompt"]
    task_id=0
    for task in config["tasks"]:
        normal_img=Image.open(config_dir.joinpath(task["normal_img"])).convert("RGB")
        reference_img=Image.open(config_dir.joinpath(task["reference_img"])).convert("RGB")
        normal_mask_img=Image.open(config_dir.joinpath(task["normal_mask_img"])).convert("L")
        reference_mask_img=Image.open(config_dir.joinpath(task["reference_mask_img"])).convert("L")
        generated_image=generator.generate_single_image(normal_img,
                                        reference_img,
                                        normal_mask_img,
                                        reference_mask_img,
                                        prompt,
                                        num_inference_steps=task["steps"],
                                        ip_scale=task["ip_scale"],
                                        strength=task["strength"])
        generated_image.save(output_dir.joinpath(f"task-{task_id}.jpg"))

if __name__ == "__main__":
    generate_anomaly()