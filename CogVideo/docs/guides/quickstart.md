# Quick Start Guide

This guide helps you get started with CogVideoX optimizations in under 5 minutes.

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/THUDM/CogVideo.git
cd CogVideo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Validate Installation

```bash
cd inference
python3 test_optimizations.py
```

Expected output:
```
✓ PASS: FPS Utils
✓ PASS: VRAM Utils
✓ PASS: Motion Presets
✓ PASS: Pipeline Utils
Total: 4/4 tests passed
🎉 All optimizations validated successfully!
```

## First Video Generation

### CLI Demo

Generate a video with optimized settings:

```bash
python3 inference/cli_demo.py \
  --prompt "A golden retriever sprinting across a meadow" \
  --model_path THUDM/CogVideoX-5b \
  --motion_preset high_motion \
  --output_path ./output.mp4
```

### Web Interface

Launch the web demo:

```bash
python3 inference/gradio_web_demo.py
```

Navigate to `http://localhost:7860` in your browser.

## Key Concepts

### Motion Presets

Choose the right preset for your use case:

- **balanced**: Default (guidance=6.0, steps=50)
- **fast**: Quick preview (guidance=5.0, steps=30)
- **quality**: Final render (guidance=7.0, steps=75)
- **high_motion**: Action/sports (guidance=6.5, steps=60) ← Golden retriever fix!
- **subtle**: Gentle motion (guidance=5.5, steps=55)

### VRAM Management

The system automatically selects the best offload strategy:

- **≥24GB VRAM** (A100, RTX 4090): Direct GPU (no offload)
- **≥16GB VRAM** (RTX 3090, 4080): Model CPU offload (2-3x faster)
- **<16GB VRAM** (RTX 3060): Sequential CPU offload (slow but works)

### FPS Auto-Detection

Videos are exported with correct FPS automatically:

- **CogVideoX** (2B, 5B, 5B-I2V): 8 fps
- **CogVideoX1.5** (5B, 5B-I2V): 16 fps

## Common Issues

### Out of Memory (OOM)

If you see OOM errors:

1. Use a smaller model: `THUDM/CogVideoX-2b`
2. Reduce frames: `--num_frames 25`
3. Enable VAE tiling (already enabled by default)

### Slow Generation

If generation is slow:

1. Check GPU is being used: `nvidia-smi`
2. Use `fast` preset for previews
3. Verify correct offload strategy with `log_vram_status()`

### Wrong FPS

FPS is now auto-detected! Remove manual `--fps` flag to use correct value.

## Next Steps

- Read [Motion Presets Guide](motion_presets.md) for detailed preset tuning
- Check [VRAM Optimization Guide](vram_optimization.md) for advanced memory management
- See [Troubleshooting](troubleshooting.md) for common issues and solutions
