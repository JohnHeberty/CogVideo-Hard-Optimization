#!/usr/bin/env python3
"""
Advanced usage examples for CogVideoX optimizations.

This script demonstrates various optimization scenarios and use cases.
"""

import sys
from pathlib import Path

# Add inference directory to path
sys.path.insert(0, str(Path(__file__).parent))

def example_1_basic_generation():
    """Example 1: Basic T2V generation with automatic optimizations."""
    print("\n" + "="*70)
    print("Example 1: Basic T2V Generation with Auto-Optimizations")
    print("="*70)
    
    print("""
This example shows the simplest way to generate a video with all
optimizations applied automatically.

Code:
    from pipeline_utils import load_pipeline
    import torch
    
    # Load with automatic optimizations
    pipe = load_pipeline(
        model_path="THUDM/CogVideoX-5b",
        pipeline_type="t2v",
        dtype=torch.bfloat16
    )
    # Automatically applies:
    # - VRAM-based offload strategy (model/sequential/none)
    # - VAE tiling/slicing
    # - DPM scheduler
    
    # Generate video
    video = pipe(
        prompt="A golden retriever running in a field",
        num_frames=49,
        num_inference_steps=50,
        guidance_scale=6.0
    ).frames[0]
    
    # Export with correct FPS
    from fps_utils import get_correct_fps
    from diffusers.utils import export_to_video
    
    fps = get_correct_fps("THUDM/CogVideoX-5b", 49)
    export_to_video(video, "output.mp4", fps=fps)

Expected VRAM (RTX 3090 24GB):
- Model offload: ~17GB during generation
- Sequential offload: ~10GB during generation (slower)
""")


def example_2_motion_presets():
    """Example 2: Using motion presets for better quality."""
    print("\n" + "="*70)
    print("Example 2: Motion Presets for Quality Control")
    print("="*70)
    
    print("""
Motion presets optimize parameters for different types of movement.
Perfect for fixing issues like the "golden retriever broken paws" problem.

Code:
    from pipeline_utils import load_pipeline
    from motion_presets import apply_preset_to_pipeline_args, get_preset
    import torch
    
    pipe = load_pipeline("THUDM/CogVideoX-5b", "t2v", torch.bfloat16)
    
    # Scenario 1: High-speed action (sports, running)
    preset = get_preset("high_motion")
    print(f"High Motion: guidance={preset.guidance_scale}, steps={preset.num_inference_steps}")
    
    base_args = {
        "prompt": "A golden retriever sprinting playfully across a meadow, ears flapping",
        "num_frames": 49
    }
    args = apply_preset_to_pipeline_args("high_motion", base_args)
    video = pipe(**args).frames[0]
    
    # Scenario 2: Gentle, subtle motion (portraits, nature)
    args_subtle = apply_preset_to_pipeline_args("subtle", {
        "prompt": "A woman's hair gently swaying in the breeze",
        "num_frames": 49
    })
    video_subtle = pipe(**args_subtle).frames[0]
    
    # Scenario 3: Fast preview (30 steps instead of 50)
    args_fast = apply_preset_to_pipeline_args("fast", {
        "prompt": "Clouds drifting across the sky",
        "num_frames": 49
    })
    video_fast = pipe(**args_fast).frames[0]  # ~40% faster

Available Presets:
- balanced: Default (guidance=6.0, steps=50)
- fast: Quick preview (guidance=5.0, steps=30) - 40% faster
- quality: Best quality (guidance=7.0, steps=75) - 50% slower
- high_motion: Action/sports (guidance=6.5, steps=60)
- subtle: Gentle motion (guidance=5.5, steps=55)
""")


def example_3_vram_monitoring():
    """Example 3: VRAM monitoring and validation."""
    print("\n" + "="*70)
    print("Example 3: VRAM Monitoring and Pre-Flight Checks")
    print("="*70)
    
    print("""
Check VRAM availability before loading models to prevent OOM crashes.

Code:
    from vram_utils import (
        get_gpu_memory_info,
        check_vram_availability,
        get_recommended_offload_strategy,
        log_vram_status,
        estimate_vram_requirement
    )
    
    # Check GPU status
    total, used, available = get_gpu_memory_info()
    print(f"GPU: {available:.1f}GB available / {total:.1f}GB total")
    
    # Check if model will fit
    model = "THUDM/CogVideoX-5b"
    is_ok, msg = check_vram_availability(model)
    print(msg)
    
    if not is_ok:
        print("Warning: Insufficient VRAM, will use sequential offload")
    
    # Get recommended strategy
    strategy = get_recommended_offload_strategy(model)
    print(f"Recommended strategy: {strategy}")
    # RTX 3090 24GB -> "model"
    # RTX 3060 12GB -> "sequential"
    
    # Monitor during generation
    log_vram_status("Before load: ")
    pipe = load_pipeline(model, "t2v")
    log_vram_status("After load: ")
    
    video = pipe(prompt="test", num_frames=49).frames[0]
    log_vram_status("After generation: ")

Output Example (RTX 3090):
    GPU: 23.7GB available / 23.7GB total
    Sufficient VRAM: 21.7GB available, 19.0GB required
    Recommended strategy: model
    Before load: VRAM Status: 0.50GB / 23.70GB used (2.1%), 23.20GB available
    After load: VRAM Status: 17.20GB / 23.70GB used (72.6%), 6.50GB available
    After generation: VRAM Status: 17.50GB / 23.70GB used (73.8%), 6.20GB available
""")


def example_4_shared_pipelines():
    """Example 4: Sharing components between pipelines."""
    print("\n" + "="*70)
    print("Example 4: Component Sharing for Multi-Pipeline Apps")
    print("="*70)
    
    print("""
When using multiple pipeline types (T2V + V2V + I2V), share components
to save ~10GB VRAM.

Code:
    from pipeline_utils import load_pipeline, load_shared_pipeline
    import torch
    
    # Load T2V first (base pipeline)
    t2v_pipe = load_pipeline(
        model_path="THUDM/CogVideoX-5b",
        pipeline_type="t2v",
        dtype=torch.bfloat16
    )
    
    # V2V shares transformer, VAE, text encoder with T2V
    v2v_pipe = load_shared_pipeline(
        model_path="THUDM/CogVideoX-5b",
        target_pipeline_type="v2v",
        base_pipeline=t2v_pipe  # Reuses components
    )
    
    # Use both pipelines
    video_t2v = t2v_pipe(
        prompt="A dog running",
        num_frames=49
    ).frames[0]
    
    from diffusers.utils import load_video
    input_video = load_video("input.mp4")[:49]
    
    video_v2v = v2v_pipe(
        prompt="A dog running in slow motion",
        video=input_video,
        num_frames=49,
        strength=0.8
    ).frames[0]

VRAM Savings:
- Without sharing: 17GB (T2V) + 17GB (V2V) = 34GB (OOM!)
- With sharing: 17GB (T2V) + 7GB (V2V shared) = 24GB (fits!)

Note: I2V uses different model (5B-I2V) so cannot share with T2V.
      Use lazy loading instead (load/unload on demand).
""")


def example_5_cli_usage():
    """Example 5: CLI usage with all features."""
    print("\n" + "="*70)
    print("Example 5: CLI Demo with All Features")
    print("="*70)
    
    print("""
The CLI demo now supports all optimizations via command-line flags.

Basic T2V with motion preset:
    python cli_demo.py \\
      --prompt "A golden retriever sprinting playfully across a meadow" \\
      --model_path THUDM/CogVideoX-5b \\
      --motion_preset high_motion \\
      --num_frames 49 \\
      --output_path golden_retriever.mp4
    
    # Automatically applies:
    # - FPS: 8 (auto-detected from model)
    # - Guidance: 6.5 (from high_motion preset)
    # - Steps: 60 (from high_motion preset)
    # - Offload: model (auto-detected for RTX 3090)

Fast preview generation:
    python cli_demo.py \\
      --prompt "Ocean waves" \\
      --model_path THUDM/CogVideoX-5b \\
      --motion_preset fast \\
      --num_frames 49
    
    # 40% faster (30 steps instead of 50)

High-quality final render:
    python cli_demo.py \\
      --prompt "A ballet dancer" \\
      --model_path THUDM/CogVideoX1.5-5b \\
      --motion_preset quality \\
      --num_frames 81
    
    # Better quality (75 steps)
    # FPS: 16 (auto-detected from 1.5 model)

Manual override (if needed):
    python cli_demo.py \\
      --prompt "Custom scene" \\
      --model_path THUDM/CogVideoX-5b \\
      --guidance_scale 7.5 \\
      --num_inference_steps 60 \\
      --fps 8 \\
      --num_frames 49
    
    # Preset NOT used, manual values applied
    # FPS validated against model (warning if mismatch)

Image-to-Video:
    python cli_demo.py \\
      --prompt "A person walking" \\
      --model_path THUDM/CogVideoX-5b-I2V \\
      --generate_type i2v \\
      --image_or_video_path input.jpg \\
      --motion_preset balanced \\
      --num_frames 49

Video-to-Video:
    python cli_demo.py \\
      --prompt "Same scene but in slow motion" \\
      --model_path THUDM/CogVideoX-5b \\
      --generate_type v2v \\
      --image_or_video_path input.mp4 \\
      --num_frames 49
""")


def example_6_gradio_demos():
    """Example 6: Gradio web demos with lazy loading."""
    print("\n" + "="*70)
    print("Example 6: Gradio Web Demos (Production Ready)")
    print("="*70)
    
    print("""
Gradio demos now use lazy loading and automatic optimizations.

Simple T2V Demo:
    python gradio_web_demo.py
    
    # Features:
    # - 0GB VRAM on startup (lazy load)
    # - Loads on first generation (~10-20s delay)
    # - Auto-detects FPS (8 or 16)
    # - Auto-selects offload strategy
    # - Caches pipeline for subsequent generations

Full-Featured Demo (T2V + I2V + V2V + Upscale):
    cd gradio_composite_demo
    python app.py
    
    # Features:
    # - 0GB VRAM on startup
    # - Lazy loads only needed pipeline per request
    # - Shares components when possible (T2V ↔ V2V)
    # - I2V uses separate model (5B-I2V)
    # - RIFE interpolation + Real-ESRGAN upscaling
    # - Auto FPS detection
    # - Auto offload strategy

VRAM Usage (RTX 3090):
    Startup:        0GB   (was 27GB - OOM!)
    T2V request:    17GB  (model offload)
    I2V request:    17GB  (model offload, separate model)
    V2V request:    17GB  (shares with T2V if loaded)
    Idle after:     0GB   (can be cached)

Performance:
    - First request: +10-20s load time
    - Cached requests: Same speed as before
    - 2-3x faster than old sequential offload
""")


def main():
    """Display all examples."""
    print("\n" + "="*70)
    print("CogVideoX Optimizations - Advanced Usage Examples")
    print("="*70)
    print("\nThese examples demonstrate various optimization scenarios.")
    print("Note: Requires diffusers, torch, and other dependencies installed.\n")
    
    examples = [
        example_1_basic_generation,
        example_2_motion_presets,
        example_3_vram_monitoring,
        example_4_shared_pipelines,
        example_5_cli_usage,
        example_6_gradio_demos,
    ]
    
    for example_func in examples:
        example_func()
    
    print("\n" + "="*70)
    print("Documentation")
    print("="*70)
    print("""
For more details, see:
- OPTIMIZATIONS.md - Complete usage guide
- test_optimizations.py - Run validation tests
- COGVIDEO_CODE_REVIEW.md - Technical analysis
- COGVIDEO_SPRINT_PLAN.md - Implementation roadmap

Quick Start:
1. Validate setup:
   python test_optimizations.py

2. Test golden retriever fix:
   python cli_demo.py --prompt "A golden retriever sprinting" \\
     --model_path THUDM/CogVideoX-5b --motion_preset high_motion

3. Launch web demo:
   python gradio_web_demo.py
""")


if __name__ == "__main__":
    main()
