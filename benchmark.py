"""
Benchmark script for Sketch-to-Material Studio
Runs generation pipeline without GUI using template sketch and prompt
"""

import os
import sys
import time
import platform
import torch
from src.pipeline import AIPipeline
from src.config import DEVICE
from src.timer import Timer


# Template configuration
TEMPLATE_SKETCH = "test/input_sketches/sample_sketch.png"
TEMPLATE_PROMPT = "detailed material texture, seamless, high quality"
OUTPUT_DIR = "test/benchmark_results"


def get_device_info():
    """Gather device and system information"""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": DEVICE,
    }
    
    if DEVICE == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    elif DEVICE == "mps":
        info["gpu_name"] = "Apple Silicon (MPS)"
    else:
        info["gpu_name"] = "CPU"
    
    return info


def main():
    print("=" * 70)
    print("  Sketch-to-Material Studio - Benchmark")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Sketch: {TEMPLATE_SKETCH}")
    print(f"Prompt: {TEMPLATE_PROMPT}")
    print("=" * 70 + "\n")
    
    # Check if sketch exists
    if not os.path.exists(TEMPLATE_SKETCH):
        print(f"ERROR: Sketch not found: {TEMPLATE_SKETCH}")
        print("Please ensure the template sketch exists.")
        sys.exit(1)
    
    # Initialize pipeline
    print("Initializing AI Pipeline...")
    init_start = time.time()
    pipeline = AIPipeline()
    init_time = time.time() - init_start
    print(f"Initialization complete ({init_time:.2f}s)\n")
    
    # Run generation
    print("Starting generation...")
    print("-" * 70)
    
    timing_log = {}
    original_timer_exit = Timer.__exit__
    
    def timer_exit_wrapper(self, exc_type, exc_val, exc_tb):
        result = original_timer_exit(self, exc_type, exc_val, exc_tb)
        timing_log[self.name] = self.elapsed
        return result
    
    Timer.__exit__ = timer_exit_wrapper
    
    overall_start = time.time()
    
    try:
        generated_files = pipeline.generate(
            prompt=TEMPLATE_PROMPT,
            init_image_path=TEMPLATE_SKETCH,
            project_name="benchmark",
            progress_callback=None
        )
        overall_time = time.time() - overall_start
        
        Timer.__exit__ = original_timer_exit
        
        # Copy results to output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        import shutil
        
        for file_path, file_type in zip(generated_files, ["albedo", "depth", "normal", "roughness"]):
            if os.path.exists(file_path):
                output_path = os.path.join(OUTPUT_DIR, f"{file_type}.png")
                shutil.copy2(file_path, output_path)
        
        # Copy input sketch
        sketch_output = os.path.join(OUTPUT_DIR, "input_sketch.png")
        shutil.copy2(TEMPLATE_SKETCH, sketch_output)
        
        # Print results
        print("-" * 70)
        print("\nBENCHMARK RESULTS")
        print("=" * 70)
        print(f"Initialization Time:     {init_time:8.2f}s")
        print(f"Sketch Processing:       {timing_log.get('Sketch Processing', 0):8.2f}s")
        print(f"Albedo Generation:       {timing_log.get('Albedo Generation', 0):8.2f}s")
        print(f"Roughness Generation:    {timing_log.get('Roughness Map Generation', 0):8.2f}s")
        print(f"Depth Estimation:        {timing_log.get('Depth Estimation', 0):8.2f}s")
        print(f"Normal Generation:       {timing_log.get('Normal Map Generation', 0):8.2f}s")
        print("-" * 70)
        print(f"TOTAL TIME:              {overall_time:8.2f}s")
        print("=" * 70)
        
        # Display device information
        device_info = get_device_info()
        print("\nDEVICE INFORMATION")
        print("=" * 70)
        print(f"Platform:                 {device_info['platform']}")
        print(f"Processor:               {device_info['processor']}")
        print(f"Python Version:          {device_info['python_version']}")
        print(f"PyTorch Version:          {device_info['pytorch_version']}")
        print(f"Device Type:              {device_info['device'].upper()}")
        print(f"GPU/Accelerator:          {device_info['gpu_name']}")
        if DEVICE == "cuda":
            print(f"CUDA Version:             {device_info.get('cuda_version', 'N/A')}")
            print(f"GPU Memory:                {device_info.get('gpu_memory', 'N/A')}")
        print("=" * 70)
        
        print(f"\nResults saved to: {OUTPUT_DIR}/")
        print("  - albedo.png")
        print("  - depth.png")
        print("  - normal.png")
        print("  - roughness.png")
        print("  - input_sketch.png")
        
    except Exception as e:
        Timer.__exit__ = original_timer_exit
        print(f"\nERROR: Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
