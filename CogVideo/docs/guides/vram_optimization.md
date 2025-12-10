# VRAM Optimization Guide

This guide explains how to minimize VRAM usage and avoid Out-of-Memory (OOM) errors when running CogVideoX.

## Understanding VRAM Requirements

### Base Requirements (Without Optimizations)

| Model | T2V | I2V | Composite Demo |
|-------|-----|-----|----------------|
| CogVideoX-2B | 8GB | 9GB | 15GB |
| CogVideoX-5B | 17GB | 18GB | **27GB** ⚠️ |
| CogVideoX1.5-5B | 20GB | 21GB | **30GB** ⚠️ |

**Problem:** RTX 3090 has 24GB → OOM on composite demo!

### With Optimizations ✅

| Model | T2V | I2V | Composite Demo (Lazy) |
|-------|-----|-----|----------------------|
| CogVideoX-2B | 6-8GB | 7-9GB | 0GB idle, 6-9GB active |
| CogVideoX-5B | 8-12GB | 10-14GB | 0GB idle, 8-14GB active |
| CogVideoX1.5-5B | 10-16GB | 12-18GB | 0GB idle, 10-18GB active |

## Optimization Strategies

### 1. Lazy Loading (Automatic)

**What:** Load models only when needed, unload after use  
**Savings:** 100% idle VRAM (0GB startup)  
**Trade-off:** ~10-20s loading time on first generation

**Enabled in:**
- `gradio_web_demo.py`
- `gradio_composite_demo/app.py`

**How it works:**
```python
from pipeline_utils import load_pipeline

# Pipeline loads on-demand
def generate_video(prompt):
    pipe = load_pipeline("THUDM/CogVideoX-5b", "t2v")  # Loads here
    video = pipe(prompt=prompt).frames[0]
    del pipe  # Unload after use
    torch.cuda.empty_cache()
    return video
```

### 2. CPU Offloading (Automatic)

**What:** Move model components between CPU and GPU dynamically  
**Savings:** 30-50% VRAM  
**Trade-off:** 2-3x slower (but still faster than sequential)

**Strategy Selection (Automatic):**

```python
from vram_utils import get_recommended_offload_strategy

strategy = get_recommended_offload_strategy("THUDM/CogVideoX-5b")
# RTX 3090 (24GB) → "model" (best balance)
# RTX 3060 (12GB) → "sequential" (slowest but works)
# A100 (40GB) → "none" (full speed)
```

**Manual Override:**
```python
from vram_utils import apply_offload_strategy

apply_offload_strategy(pipe, strategy="model")
# Options: "none", "model", "sequential"
```

### 3. VAE Tiling (Automatic)

**What:** Process video in tiles instead of all-at-once  
**Savings:** 20-30% VRAM during VAE decode  
**Trade-off:** Minimal quality impact

**Enabled by default:**
```python
from vram_utils import configure_vae_tiling

configure_vae_tiling(pipe, enable=True)
```

**Advanced tuning:**
```python
configure_vae_tiling(
    pipe,
    enable=True,
    tile_sample_min_height=256,  # Smaller = less VRAM
    tile_sample_min_width=256
)
```

### 4. Component Sharing

**What:** Reuse transformer/VAE across multiple pipelines  
**Savings:** ~10GB when using multiple pipelines  
**Trade-off:** None!

**Example (T2V + V2V):**
```python
from pipeline_utils import load_shared_pipeline

# Load T2V first
pipe_t2v = load_pipeline("THUDM/CogVideoX-5b", "t2v")

# V2V shares components
pipe_v2v = load_shared_pipeline(
    "THUDM/CogVideoX-5b",
    "v2v",
    shared_components={
        "transformer": pipe_t2v.transformer,
        "vae": pipe_t2v.vae
    }
)
```

## Monitoring VRAM

### Real-time Monitoring

```bash
# Terminal 1: Run generation
python3 inference/cli_demo.py --prompt "..."

# Terminal 2: Monitor VRAM
watch -n 1 nvidia-smi
```

### Programmatic Monitoring

```python
from vram_utils import get_gpu_memory_info, log_vram_status

# Get current VRAM
total_gb, used_gb, available_gb = get_gpu_memory_info()
print(f"{available_gb:.1f}GB available")

# Pretty-print status
log_vram_status()
# Output: VRAM Status: 8.20GB / 23.70GB used (35%), 15.50GB available
```

### Check Before Loading

```python
from vram_utils import check_vram_availability

is_sufficient, message = check_vram_availability(
    "THUDM/CogVideoX-5b",
    required_buffer_gb=2.0
)

if not is_sufficient:
    print(f"⚠️ {message}")
    # Fallback: use 2B model or reduce frames
```

## Troubleshooting OOM Errors

### Scenario 1: OOM During Model Load

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.34 GiB
```

**Solutions:**

1. **Use smaller model:**
   ```bash
   --model_path THUDM/CogVideoX-2b  # Instead of 5b
   ```

2. **Enable sequential offload:**
   ```python
   apply_offload_strategy(pipe, strategy="sequential")
   ```

3. **Close other GPU programs:**
   ```bash
   nvidia-smi  # Check what's using GPU
   kill <pid>  # Kill unnecessary processes
   ```

### Scenario 2: OOM During Generation

**Symptoms:**
- Loads fine, crashes during `pipe(prompt=...)`

**Solutions:**

1. **Reduce frames:**
   ```bash
   --num_frames 25  # Instead of 49
   ```

2. **Enable VAE tiling:**
   ```python
   configure_vae_tiling(pipe, enable=True)
   ```

3. **Lower resolution (I2V only):**
   ```python
   image = image.resize((480, 320))  # Instead of 720x480
   ```

### Scenario 3: OOM in Composite Demo

**Symptoms:**
- `gradio_composite_demo/app.py` crashes on startup

**Solutions:**

1. **Lazy loading already enabled** - if still failing:
   ```python
   # Edit app.py: Reduce cache size
   MAX_CACHE_SIZE = 1  # Only keep 1 pipeline loaded
   ```

2. **Use smaller models:**
   ```python
   DEFAULT_MODEL_T2V = "THUDM/CogVideoX-2b"
   ```

3. **Disable V2V or upscale** to reduce pipeline count

## Best Practices

### Development Workflow

1. **Start with `fast` preset:**
   ```bash
   --motion_preset fast  # Quick iterations
   ```

2. **Use 2B model for testing:**
   ```bash
   --model_path THUDM/CogVideoX-2b
   ```

3. **Monitor VRAM throughout:**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Final render with 5B + `quality`:**
   ```bash
   --model_path THUDM/CogVideoX-5b --motion_preset quality
   ```

### Production Deployment

1. **Always use lazy loading** (enabled by default in demos)
2. **Set VRAM buffer:**
   ```python
   check_vram_availability(model_path, required_buffer_gb=3.0)
   ```
3. **Implement fallback:**
   ```python
   try:
       pipe = load_pipeline("THUDM/CogVideoX-5b", "t2v")
   except RuntimeError:  # OOM
       pipe = load_pipeline("THUDM/CogVideoX-2b", "t2v")
   ```

## Hardware Recommendations

### GPU Requirements

| GPU | VRAM | Recommended Model | Notes |
|-----|------|-------------------|-------|
| RTX 3060 | 12GB | CogVideoX-2B | Use sequential offload |
| RTX 3070 | 8GB | CogVideoX-2B (25 frames) | Very limited |
| RTX 3080 | 10GB | CogVideoX-2B | Comfortable |
| RTX 3090 | 24GB | CogVideoX-5B ✅ | **Perfect!** |
| RTX 4080 | 16GB | CogVideoX-5B | Use model offload |
| RTX 4090 | 24GB | CogVideoX1.5-5B ✅ | Excellent |
| A100 | 40GB | Any model | Full speed |

### RAM Requirements

- **Minimum:** 16GB system RAM
- **Recommended:** 32GB (for model offloading)
- **Optimal:** 64GB (for multiple pipelines)

## Advanced Techniques

### Dynamic Frame Count

Adjust frames based on available VRAM:

```python
from vram_utils import get_gpu_memory_info

total_gb, used_gb, available_gb = get_gpu_memory_info()

if available_gb >= 20:
    num_frames = 81  # CogVideoX1.5
elif available_gb >= 14:
    num_frames = 49  # CogVideoX
else:
    num_frames = 25  # Reduced
```

### Gradient Checkpointing

For extreme VRAM savings (not implemented yet):

```python
pipe.transformer.enable_gradient_checkpointing()
# Saves ~30% VRAM but 40% slower
```

### Mixed Precision

Already enabled by default:

```python
pipe = CogVideoXPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16  # Uses less VRAM than float32
)
```

## Benchmarking

Test your setup:

```bash
cd inference
python3 -c "
from vram_utils import log_vram_status
from pipeline_utils import load_pipeline

log_vram_status()
print('Loading pipeline...')
pipe = load_pipeline('THUDM/CogVideoX-5b', 't2v')
log_vram_status()
"
```

Expected output (RTX 3090):
```
VRAM Status: 0.50GB / 23.70GB used (2%), 23.20GB available
Loading pipeline...
VRAM Status: 11.20GB / 23.70GB used (47%), 12.50GB available
```

## Next Steps

- Read [Motion Presets Guide](motion_presets.md) for quality tuning
- Check [API Reference](../api/vram_utils.rst) for full API
- See [Troubleshooting Guide](troubleshooting.md) for more solutions
