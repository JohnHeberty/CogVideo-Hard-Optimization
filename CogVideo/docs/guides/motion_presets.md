# Motion Presets Guide

Motion presets provide pre-configured settings for different types of video generation,
optimizing the balance between quality, speed, and motion characteristics.

## Available Presets

### 1. Balanced (Default)

**Use Case:** General-purpose video generation

**Settings:**
- Guidance Scale: 6.0
- Inference Steps: 50

**Best For:**
- Everyday use
- Mixed motion scenes
- When you're not sure which preset to use

**Example:**
```bash
python3 cli_demo.py \
  --prompt "A cat walking on a fence" \
  --model_path THUDM/CogVideoX-5b \
  --motion_preset balanced
```

### 2. Fast (Preview Mode)

**Use Case:** Quick previews and iteration

**Settings:**
- Guidance Scale: 5.0
- Inference Steps: 30

**Performance:** ~40% faster than balanced

**Best For:**
- Testing prompts
- Quick previews
- Iterating on ideas
- Resource-constrained environments

**Trade-offs:** Lower quality, less coherent motion

**Example:**
```bash
python3 cli_demo.py \
  --prompt "Ocean waves" \
  --model_path THUDM/CogVideoX-5b \
  --motion_preset fast
```

### 3. Quality (Final Render)

**Use Case:** Production-quality output

**Settings:**
- Guidance Scale: 7.0
- Inference Steps: 75

**Performance:** ~50% slower than balanced

**Best For:**
- Final renders
- Client deliverables
- High-quality demos
- Slow, deliberate motion

**Trade-offs:** Slower generation time

**Example:**
```bash
python3 cli_demo.py \
  --prompt "Sunset over mountains, time-lapse" \
  --model_path THUDM/CogVideoX-5b \
  --motion_preset quality
```

### 4. High Motion (Action/Sports) ⭐

**Use Case:** Fast-paced action and complex motion

**Settings:**
- Guidance Scale: 6.5
- Inference Steps: 60

**Best For:**
- **Running/sprinting animals** (fixes "golden retriever" issue!)
- Sports and athletics
- Fast camera movement
- Complex multi-object motion

**Why It Works:**
Higher guidance scale (6.5) keeps the model focused during rapid motion,
preventing artifacts like "broken limbs" or motion blur.

**Example (Golden Retriever Fix):**
```bash
python3 cli_demo.py \
  --prompt "A golden retriever sprinting playfully across a meadow, ears flapping" \
  --model_path THUDM/CogVideoX-5b \
  --motion_preset high_motion
```

**Other Good Uses:**
- "A cheetah chasing prey"
- "Basketball player dunking"
- "FPV drone racing through forest"
- "Parkour athlete jumping between rooftops"

### 5. Subtle (Gentle Motion)

**Use Case:** Slow, gentle movements

**Settings:**
- Guidance Scale: 5.5
- Inference Steps: 55

**Best For:**
- Portrait videos (slight head turn)
- Floating/drifting objects
- Ambient scenes
- Minimalist motion

**Example:**
```bash
python3 cli_demo.py \
  --prompt "A flower slowly blooming" \
  --model_path THUDM/CogVideoX-5b \
  --motion_preset subtle
```

## Choosing the Right Preset

### Decision Tree

```
Is this for final production?
├─ Yes → quality
└─ No
    └─ Need fast iteration?
        ├─ Yes → fast
        └─ No
            └─ What kind of motion?
                ├─ Fast action/sports → high_motion
                ├─ Gentle/slow → subtle
                └─ Mixed/unsure → balanced
```

### Motion Type Matrix

| Motion Type | Recommended Preset | Alternative |
|-------------|-------------------|-------------|
| Running animals | **high_motion** | balanced |
| Sports/athletics | **high_motion** | balanced |
| Walking | balanced | subtle |
| Portraits | subtle | balanced |
| Camera pan/zoom | balanced | high_motion |
| Time-lapse | quality | balanced |
| Floating objects | subtle | balanced |
| Action scenes | high_motion | quality |

## Programmatic Usage

### Python API

```python
from motion_presets import get_preset, apply_preset_to_pipeline_args

# Get preset details
preset = get_preset("high_motion")
print(preset.guidance_scale)  # 6.5
print(preset.num_inference_steps)  # 60

# Apply to pipeline arguments
args = apply_preset_to_pipeline_args("high_motion", {
    "prompt": "A golden retriever sprinting",
    "num_frames": 49,
    "seed": 42
})

# Generate video with preset
video = pipe(**args).frames[0]
```

### List All Presets

```python
from motion_presets import list_presets

presets = list_presets()
for name, preset in presets:
    print(f"{name}: guidance={preset.guidance_scale}, steps={preset.num_inference_steps}")
```

## Advanced: Custom Presets

You can create custom presets by modifying `motion_presets.py`:

```python
MOTION_PRESETS = {
    # ... existing presets ...
    
    "my_custom": MotionPreset(
        name="my_custom",
        description="My custom settings",
        guidance_scale=6.3,
        num_inference_steps=55,
        use_dynamic_cfg=True
    )
}
```

## Performance Comparison

Benchmark on RTX 3090 24GB (CogVideoX-5b, 49 frames):

| Preset | Time | Quality | VRAM |
|--------|------|---------|------|
| fast | ~2min | ⭐⭐⭐ | 8-10GB |
| balanced | ~3.5min | ⭐⭐⭐⭐ | 8-12GB |
| subtle | ~3.8min | ⭐⭐⭐⭐ | 8-12GB |
| high_motion | ~4min | ⭐⭐⭐⭐ | 10-14GB |
| quality | ~5min | ⭐⭐⭐⭐⭐ | 10-14GB |

## Troubleshooting

### Motion is Still Broken

If `high_motion` doesn't fix motion artifacts:

1. **Check prompt:** Be specific about the motion
   - ❌ "A dog"
   - ✅ "A golden retriever sprinting with smooth gait"

2. **Try different seed:** Some seeds work better
   ```bash
   --seed 42  # or 123, 456, etc.
   ```

3. **Increase frames:** More frames = smoother motion
   ```bash
   --num_frames 81  # if using CogVideoX1.5
   ```

4. **Use I2V:** Start with a good reference image
   ```bash
   --image_path ./dog_running.jpg
   ```

### Quality is Too Low

If `quality` preset isn't good enough:

1. **Increase steps manually:**
   ```bash
   --num_inference_steps 100
   ```

2. **Adjust guidance:**
   ```bash
   --guidance_scale 7.5
   ```

3. **Use larger model:**
   ```bash
   --model_path THUDM/CogVideoX1.5-5b
   ```

## Best Practices

1. **Start with `fast`** for prompt iteration
2. **Use `high_motion`** for any fast-moving subjects
3. **Use `quality`** only for final renders
4. **Combine with good prompts** - presets can't fix vague prompts
5. **Monitor VRAM** - higher quality = more VRAM

## Next Steps

- Read [VRAM Optimization Guide](vram_optimization.md)
- Check [Troubleshooting Guide](troubleshooting.md)
- See [API Reference](../api/motion_presets.rst) for full details
