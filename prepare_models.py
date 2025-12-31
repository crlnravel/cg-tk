#!/usr/bin/env python3
"""
Pre-download script for AI models.
Run this once before using the main application to download all required models.
This prevents downloading models every time you run the application.
"""

import os
import sys
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from transformers import pipeline as depth_pipeline

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

print("=" * 60)
print("AI Models Pre-Download Script")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.backends.mps.is_available():
    print(f"MPS Available: {torch.backends.mps.is_available()}")
print("=" * 60)
print()

# Models to download
MODELS = [
    {
        "name": "ControlNet (Scribble)",
        "model_id": "lllyasviel/sd-controlnet-scribble",
        "type": "controlnet"
    },
    {
        "name": "Stable Diffusion v1.5",
        "model_id": "runwayml/stable-diffusion-v1-5",
        "type": "sd"
    },
    {
        "name": "Depth Estimation (Depth-Anything-Small)",
        "model_id": "LiheYoung/depth-anything-small-hf",
        "type": "depth"
    }
]

def download_controlnet(model_id, device):
    """Download ControlNet model"""
    print(f"Downloading ControlNet: {model_id}")
    print("This may take several minutes...")
    try:
        model = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        print(f"✓ ControlNet downloaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error downloading ControlNet: {e}")
        return None

def download_stable_diffusion(model_id, controlnet, device):
    """Download Stable Diffusion pipeline"""
    print(f"Downloading Stable Diffusion: {model_id}")
    print("This may take several minutes...")
    try:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        print(f"✓ Stable Diffusion downloaded successfully")
        return pipe
    except Exception as e:
        print(f"✗ Error downloading Stable Diffusion: {e}")
        return None

def download_depth_estimator(model_id, device):
    """Download depth estimation model"""
    print(f"Downloading Depth Estimator: {model_id}")
    print("This may take several minutes...")
    try:
        depth_pipe = depth_pipeline(
            task="depth-estimation",
            model=model_id,
            device=device,
        )
        print(f"✓ Depth Estimator downloaded successfully")
        return depth_pipe
    except Exception as e:
        print(f"✗ Error downloading Depth Estimator: {e}")
        return None

def main():
    print("Starting model downloads...")
    print("Note: Models are cached in ~/.cache/huggingface/")
    print("They will not be re-downloaded on subsequent runs.")
    print()
    
    success_count = 0
    total_models = len(MODELS)
    
    # Download ControlNet
    print(f"[1/{total_models}] {MODELS[0]['name']}")
    controlnet = download_controlnet(MODELS[0]['model_id'], DEVICE)
    if controlnet:
        success_count += 1
    print()
    
    # Download Stable Diffusion (requires ControlNet)
    if controlnet:
        print(f"[2/{total_models}] {MODELS[1]['name']}")
        pipe = download_stable_diffusion(MODELS[1]['model_id'], controlnet, DEVICE)
        if pipe:
            success_count += 1
        # Clean up to save memory
        del pipe
        del controlnet
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
    else:
        print(f"[2/{total_models}] {MODELS[1]['name']} - SKIPPED (ControlNet failed)")
    print()
    
    # Download Depth Estimator
    print(f"[3/{total_models}] {MODELS[2]['name']}")
    depth_pipe = download_depth_estimator(MODELS[2]['model_id'], DEVICE)
    if depth_pipe:
        success_count += 1
    # Clean up
    del depth_pipe
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    print()
    
    # Summary
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successfully downloaded: {success_count}/{total_models} models")
    
    if success_count == total_models:
        print("✓ All models downloaded successfully!")
        print("You can now run 'python main_sketch.py' without downloading.")
        print()
        print("Models are cached in:")
        cache_dir = os.path.expanduser("~/.cache/huggingface/")
        print(f"  {cache_dir}")
    else:
        print("⚠ Some models failed to download.")
        print("Please check your internet connection and try again.")
        print("You can re-run this script to retry failed downloads.")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Partially downloaded models are cached and will resume on next run.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)

