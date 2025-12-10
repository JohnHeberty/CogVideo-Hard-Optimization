# Troubleshooting Guide

Common issues and solutions when using CogVideoX optimizations.

## Installation Issues

### ImportError: No module named 'diffusers'

**Problem:**
```
ImportError: No module named 'diffusers'
```

**Solution:**
```bash
pip install diffusers>=0.35.2
# or
pip install -r CogVideo/requirements.txt
```

### CUDA Not Available

**Problem:**
```
RuntimeError: Torch not compiled with CUDA enabled
```

**Solutions:**

1. **Install CUDA-enabled PyTorch:**
   ```bash
   pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Verify CUDA:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # e.g., "NVIDIA RTX 3090"
   ```

## Generation Issues

### Out of Memory (OOM)

**Problem:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Diagnosis:**
```python
from vram_utils import get_gpu_memory_info, log_vram_status

log_vram_status()
```

**Solutions (in order of preference):**

1. **Use smaller model:**
   ```bash
   --model_path THUDM/CogVideoX-2b
   ```

2. **Reduce frames:**
   ```bash
   --num_frames 25  # Instead of 49
   ```

3. **Enable sequential offload:**
   ```python
   from vram_utils import apply_offload_strategy
   apply_offload_strategy(pipe, strategy="sequential")
   ```

4. **Close other GPU programs:**
   ```bash
   nvidia-smi  # See what's running
   kill <pid>  # Kill unnecessary processes
   ```

### Video Has Wrong Timing/Speed

**Problem:**
Video plays too fast or too slow

**Diagnosis:**
```python
from fps_utils import get_correct_fps

fps = get_correct_fps("THUDM/CogVideoX-5b", 49)
print(fps)  # Should be 8
```

**Solution:**
Remove manual `--fps` flag to use auto-detection:
```bash
# ❌ Wrong
--fps 16

# ✅ Correct (auto-detect)
# Just omit --fps flag
```

### Motion Artifacts (Broken Limbs, Jitter)

**Problem:**
Fast-moving subjects have broken/distorted limbs (e.g., "golden retriever")

**Solution:**
Use `high_motion` preset:
```bash
--motion_preset high_motion
```

**If still broken:**

1. **Improve prompt specificity:**
   ```
   # ❌ Vague
   "A dog running"
   
   # ✅ Specific
   "A golden retriever sprinting gracefully with smooth gait, ears flapping"
   ```

2. **Try different seeds:**
   ```bash
   --seed 42  # Try 42, 123, 456, etc.
   ```

3. **Use more frames (if CogVideoX1.5):**
   ```bash
   --num_frames 81  # More frames = smoother
   ```

### Generation is Very Slow

**Problem:**
Taking 10+ minutes for 49 frames

**Diagnosis:**
```python
from vram_utils import get_recommended_offload_strategy

strategy = get_recommended_offload_strategy("THUDM/CogVideoX-5b")
print(strategy)  # Should be "model" on RTX 3090, not "sequential"
```

**Solutions:**

1. **Check offload strategy:**
   ```python
   # Ensure you're not using sequential offload
   apply_offload_strategy(pipe, strategy="model")  # Faster
   ```

2. **Use `fast` preset:**
   ```bash
   --motion_preset fast  # 40% faster
   ```

3. **Verify GPU usage:**
   ```bash
   nvidia-smi  # GPU should be 90-100%
   ```

## Demo Issues

### Gradio Web Demo Won't Start

**Problem:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Kill existing process
lsof -ti:7860 | xargs kill -9

# Or use different port
python3 gradio_web_demo.py --server-port 7861
```

### Composite Demo Crashes on Startup

**Problem:**
OOM crash when starting `gradio_composite_demo/app.py`

**Diagnosis:**
Lazy loading might not be enabled

**Solution:**
Check `app.py`:
```python
# Should have lazy loading (COGVIDEOX_PIPELINES = {})
COGVIDEOX_PIPELINES = {}  # ✅ Correct (lazy)

# NOT this:
# pipe_text = load_pipeline(...)  # ❌ Wrong (eager load)
```

### First Generation is Slow

**Problem:**
First video takes 2-3x longer than subsequent ones

**Explanation:**
This is **expected behavior** with lazy loading!

- **First generation:** ~20s model load + ~3.5min generation = ~4min total
- **Subsequent:** ~3.5min (model cached)

**Not an issue** - this saves VRAM when idle.

## API Issues

### MotionPreset Object Not Subscriptable

**Problem:**
```python
preset = get_preset("high_motion")
print(preset["guidance_scale"])  # Error!
```

**Solution:**
Use attribute access, not dict access:
```python
preset = get_preset("high_motion")
print(preset.guidance_scale)  # ✅ Correct
```

### Pipeline Loading Fails

**Problem:**
```
ValueError: Invalid model path: my-custom-model
```

**Diagnosis:**
```python
from pipeline_utils import validate_model_path

try:
    validate_model_path("my-custom-model")
except ValueError as e:
    print(e)  # Shows which models are supported
```

**Solution:**
Use supported models:
- `THUDM/CogVideoX-2b`
- `THUDM/CogVideoX-5b`
- `THUDM/CogVideoX-5b-I2V`
- `THUDM/CogVideoX1.5-5b`
- `THUDM/CogVideoX1.5-5b-I2V`

## Docker Issues

### Container Crashes with OOM

**Problem:**
Docker container exits with OOM

**Solution:**
```yaml
# docker-compose.yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
    limits:
      memory: 32G  # Increase if needed
```

### GPU Not Available in Container

**Problem:**
```
RuntimeError: CUDA not available in container
```

**Solution:**
```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Quality Issues

### Video is Blurry/Low Quality

**Solutions:**

1. **Use `quality` preset:**
   ```bash
   --motion_preset quality
   ```

2. **Increase inference steps:**
   ```bash
   --num_inference_steps 75  # Or higher
   ```

3. **Use larger model:**
   ```bash
   --model_path THUDM/CogVideoX1.5-5b  # Instead of 2b
   ```

4. **Improve prompt:**
   ```
   # Add quality keywords
   "high quality, detailed, sharp, 4k"
   ```

### Colors Look Wrong

**Problem:**
Video has weird color shifts

**Potential Causes:**

1. **VAE precision issue** - ensure using bfloat16:
   ```python
   pipe = CogVideoXPipeline.from_pretrained(
       model_path,
       torch_dtype=torch.bfloat16  # Not float16
   )
   ```

2. **Prompt issue:**
   ```
   # Specify desired colors
   "vibrant colors" or "natural lighting"
   ```

## Getting Help

### Collect Debug Information

Before asking for help, run:

```bash
cd inference
python3 -c "
import sys
import torch
from vram_utils import log_vram_status

print('=== System Info ===')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    log_vram_status()
print()

from fps_utils import get_correct_fps
print('=== FPS Utils Test ===')
print(f'CogVideoX-5b @ 49 frames: {get_correct_fps(\"THUDM/CogVideoX-5b\", 49)}fps')
print()

from motion_presets import list_presets
print('=== Available Presets ===')
for name, _ in list_presets():
    print(f'  - {name}')
"
```

### Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `CUDA out of memory` | Insufficient VRAM | Use smaller model/fewer frames |
| `No module named...` | Missing dependency | `pip install -r requirements.txt` |
| `Address already in use` | Port conflict | Kill process or use `--server-port` |
| `Invalid model path` | Unsupported model | Use THUDM/CogVideoX-* models |
| `FPS mismatch` | Manual FPS override | Remove `--fps` flag |

### Still Stuck?

1. **Check validation report:**
   ```bash
   cat VALIDATION_REPORT.md
   ```

2. **Run tests:**
   ```bash
   cd inference
   python3 test_optimizations.py
   ```

3. **Review logs:**
   - Enable verbose mode: `LOG_LEVEL=DEBUG python3 cli_demo.py ...`
   - Check terminal output for errors

4. **File an issue:**
   - Include system info (from debug script above)
   - Include full error message and traceback
   - Describe steps to reproduce

## Next Steps

- Read [Quick Start Guide](quickstart.md) for basic usage
- Check [Motion Presets Guide](motion_presets.md) for quality tuning
- See [VRAM Optimization Guide](vram_optimization.md) for memory management
