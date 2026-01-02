import os
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from transformers import pipeline as depth_pipeline

from src.config import DEVICE
from src.timer import Timer


class AIPipeline:
    def __init__(self):
        print(f"Initializing AI on {DEVICE}")
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(DEVICE)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        if DEVICE == "cuda":
            self.pipe.enable_model_cpu_offload()

        self.depth_pipe = depth_pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=DEVICE,
        )

    def generate(self, prompt, init_image_path, project_name, progress_callback=None):
        overall_timer = Timer("OVERALL GENERATION")
        overall_timer.__enter__()

        project_path = os.path.join("projects", project_name)
        os.makedirs(project_path, exist_ok=True)

        if progress_callback:
            progress_callback(0.1, "Processing Sketch...")
        with Timer("Sketch Processing"):
            sketch = Image.open(init_image_path).convert("RGB").resize((512, 512))
            sketch_gray = sketch.convert("L")
            sketch_array = np.array(sketch_gray)
            _, binary = cv2.threshold(sketch_array, 127, 255, cv2.THRESH_BINARY)
            sketch = Image.fromarray(binary).convert("RGB")
            sketch = ImageOps.invert(sketch)
            sketch.save(os.path.join(project_path, "sketch_processed.png"))

        if progress_callback:
            progress_callback(0.2, "Generating Albedo...")

        pos_prompt = f"{prompt}, seamless texture, top down view, flat lighting, 8k, highly detailed, photorealistic material"
        neg_prompt = "perspective, 3d, shadows, people, objects, text, watermark, blurry, low quality, distortion"

        def pipe_callback(step, timestep, latents):
            if progress_callback:
                prog = 0.2 + (step / 30) * 0.5
                progress_callback(prog, f"Diffusion Step {step}/30")

        with Timer("Albedo Generation"):
            albedo = self.pipe(
                pos_prompt,
                image=sketch,
                negative_prompt=neg_prompt,
                num_inference_steps=30,
                controlnet_conditioning_scale=0.7,
                guidance_scale=8.0,
                callback=pipe_callback,
                callback_steps=1,
            ).images[0]

        p_albedo = os.path.join(project_path, "albedo.png")
        albedo.save(p_albedo)

        if progress_callback:
            progress_callback(0.8, "Generating Roughness...")
        with Timer("Roughness Map Generation"):
            img_cv = cv2.cvtColor(np.array(albedo), cv2.COLOR_RGB2GRAY)
            roughness = cv2.normalize(img_cv, None, 0, 255, cv2.NORM_MINMAX)
            roughness = 255 - roughness
            p_rough = os.path.join(project_path, "roughness.png")
            Image.fromarray(roughness).save(p_rough)

        if progress_callback:
            progress_callback(0.9, "Calculating Depth...")
        with Timer("Depth Estimation"):
            depth_data = self.depth_pipe(albedo)["depth"].resize((512, 512))
            depth_np = np.array(depth_data)
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max > d_min:
                depth_np = ((depth_np - d_min) / (d_max - d_min)) * 255.0
            p_depth = os.path.join(project_path, "depth.png")
            Image.fromarray(depth_np.astype(np.uint8)).save(p_depth)

        if progress_callback:
            progress_callback(0.95, "Calculating Normals...")
        with Timer("Normal Map Generation"):
            d_np = depth_np.astype(np.float32)
            sobelx = cv2.Sobel(d_np, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(d_np, cv2.CV_64F, 0, 1, ksize=5)
            normal = np.dstack((-sobelx, -sobely, np.ones_like(d_np) * 500.0))
            norm = np.linalg.norm(normal, axis=2, keepdims=True)
            normal = ((normal / norm) * 0.5 + 0.5) * 255
            p_normal = os.path.join(project_path, "normal.png")
            Image.fromarray(normal.astype(np.uint8)).save(p_normal)

        if progress_callback:
            progress_callback(1.0, "Done!")

        overall_timer.__exit__(None, None, None)
        return (p_albedo, p_depth, p_normal, p_rough)

