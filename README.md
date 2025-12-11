# 🎬 CogVideoX RTX 3090 - Production Ready

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![VRAM](https://img.shields.io/badge/VRAM-24GB%20RTX%203090-orange)](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)
[![Memory](https://img.shields.io/badge/Auto%20Cleanup-Memory%20Manager-brightgreen)](#-memory-management)

> **Production-ready CogVideoX optimized for NVIDIA RTX 3090 with automatic memory management and lazy loading**

Complete video generation solution with FPS auto-detection, VRAM optimization, motion presets, H.264 codec, intelligent memory management for multi-service environments.

---

## 🎥 Example Output

**Input Image** → **Prompt** → **Generated Video**

<table>
<tr>
<td width="33%" align="center">
<img src="movie/images.jpg" alt="Input: Motorcycle racing scene" width="250"/>
<br/><sub><b>Input Image</b></sub>
</td>
<td width="33%" align="center">
<pre><code>racing motorcycles 
on the road</code></pre>
<br/><sub><b>Prompt Used</b></sub>
</td>
<td width="33%" align="center">
<a href="movie/Teste.mp4"><img src="movie/images.jpg" alt="Video thumbnail" width="250"/></a>
<br/><sub><b>Generated Video</b></sub>
<br/><sub>📹 <a href="movie/Teste.mp4">View Teste.mp4</a></sub>
</td>
</tr>
</table>

**Generation Stats:**
- 🎬 Model: `CogVideoX-5b-I2V` (Image-to-Video)
- 📐 Resolution: 720x480 pixels
- ⏱️ Duration: 6 seconds (49 frames @ 8fps)
- 🎮 VRAM Peak: 18GB / 24GB (75%)
- ⚡ Generation Time: ~600 seconds on RTX 3090
- 💾 File Size: 358KB (H.264 optimized)

---

## 🚀 Key Features

## ✨ Key Features

### 🧠 Memory Management (Auto-Cleanup)
- **Automatic Model Unloading** - Models are freed immediately after use
- **Context Managers** - Safe resource management with `with` blocks
- **Multi-Service Ready** - VRAM/RAM freed for other microservices
- **Aggressive Cleanup** - Production mode with automatic garbage collection
- **Real-time Monitoring** - RAM and VRAM usage tracking

### 🎯 Core Optimizations
- ⚡ **Lazy Loading** - Models load on-demand only when needed
- 🎞️ **FPS Auto-Detection** - Automatically detects 8fps (CogVideoX) or 16fps (CogVideoX1.5)
- 💾 **VRAM Management** - Smart offload strategies prevent OOM errors
- 🎬 **Motion Presets** - 5 quality levels (fixes motion artifacts)
- 📹 **H.264 Codec** - Optimized video export with 85-95% file size reduction

### 🛠️ Advanced Features
- 🐳 **Docker Ready** - Complete containerization with NVIDIA GPU support
- 🔄 **Error Recovery** - Automatic cleanup on failures with friendly error messages
- 📊 **Structured Logging** - Real-time insights with colors and timing
- 🎨 **Gradio UI** - Modern web interface for Text-to-Video, Image-to-Video, and Video-to-Video

### 💾 Memory Usage

| Stage | RAM Usage | VRAM Usage | Status |
|-------|-----------|------------|--------|
| **Idle (Waiting)** | ~700MB | ~0GB | ✅ Excellent |
| **Generating Video** | 3-5GB | 15-18GB | ✅ Optimal |
| **With Upscale** | 15GB peak | 18-20GB | ⚠️ Auto-unload after use |
| **With Interpolation** | 12GB peak | 16-18GB | ⚠️ Auto-unload after use |

**Memory Manager ensures models are unloaded after each operation, keeping idle state minimal.**

---

## 📦 Installation

### Prerequisites

- **GPU:** NVIDIA RTX 3090 (24GB VRAM) or similar
- **RAM:** 32GB recommended (minimum 17GB)
- **Disk:** 70GB free space (for models)
- **OS:** Linux (Ubuntu 20.04+, Debian 11+) or Windows with WSL2
- **Docker:** 20.10+ with NVIDIA Container Toolkit

### Quick Start (Docker - Recommended)

```bash
# 1. Clone repository
git clone https://github.com/JohnHeberty/CogVideo-Hard-Optimization.git
cd CogVideo-Hard-Optimization

# 2. Start container (downloads models automatically on first run)
docker compose up -d

# 3. Check logs
docker logs cogvideo -f

# 4. Access Gradio UI at http://localhost:7860
```

**First Run Notes:**
- ⏳ Models download automatically (~55GB total)
- 💾 Downloads go to `./data/huggingface` (persisted)
- 📹 Generated videos save to `./data/output`
- ⏱️ First startup takes 2-3 minutes

### Manual Installation (Advanced)

```bash
# 1. Clone repository
git clone https://github.com/JohnHeberty/CogVideo-Hard-Optimization.git
cd CogVideo-Hard-Optimization

# 2. Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r CogVideo/inference/gradio_composite_demo/requirements.txt
pip install imageio imageio-ffmpeg  # For H.264 support

# 4. Run application
cd CogVideo/inference/gradio_composite_demo
python app.py

# 5. Access at http://localhost:7860
```

---

## 🎨 Usage Examples

### Text-to-Video (T2V)

1. Open Gradio UI at `http://localhost:7860`
2. Select **"Text to Video"** tab
3. Enter prompt: `"A golden retriever running on a beach at sunset"`
4. Click **Generate**
5. Video saves to `./data/output/`

**Recommended Settings:**
- Model: `CogVideoX-5b` (best quality)
- Steps: 50-60 (balance quality/speed)
- Guidance: 6.0-7.0
- Motion Preset: `high_motion` (for action scenes)

### Image-to-Video (I2V)

1. Select **"Image to Video"** tab
2. Upload your image (example: `movie/images.jpg`)
3. Enter prompt: `"racing motorcycles on the road"`
4. Click **Generate**
5. Result saves to `./data/output/`

**Tips:**
- Use clear, detailed prompts
- Image resolution: 720x480 or 1280x720
- Avoid overly complex scenes for first tests

---

## 📁 Project Structure

```
CogVideo-Hard-Optimization/
├── CogVideo/                          # Main source code
│   ├── inference/
│   │   ├── gradio_composite_demo/    # Web UI application
│   │   │   ├── app.py               # Main Gradio app (lazy loading)
│   │   │   ├── utils.py             # Utility functions
│   │   │   └── output/              # Generated videos (bind to ./data/output)
│   │   ├── fps_utils.py             # FPS auto-detection
│   │   ├── vram_utils.py            # VRAM management
│   │   ├── motion_presets.py        # Motion quality presets
│   │   ├── pipeline_utils.py        # Pipeline loading
│   │   ├── video_export_utils.py    # H.264 codec export
│   │   └── logging_config.py        # Structured logging
│   ├── tests/                        # 77 unit tests
│   └── docs/                         # Sphinx documentation
├── data/
│   ├── huggingface/                  # Downloaded models (55GB)
│   ├── output/                       # Generated videos
│   └── gradio_tmp/                   # Temporary files
├── movie/                             # Example inputs/outputs
│   ├── images.jpg                    # Example input image
│   └── Teste.mp4                     # Example generated video
├── docker-compose.yaml               # Docker configuration
├── Dockerfile                        # Container build instructions
├── AUDIT.md                          # System audit report
└── README.md                         # This file
```

---

## 🐳 Docker Configuration

### Environment Variables

```yaml
# Models (can be changed)
COGVIDEO_T2V_MODEL=THUDM/CogVideoX-5b          # Text-to-Video model
COGVIDEO_I2V_MODEL=THUDM/CogVideoX-5b-I2V      # Image-to-Video model

# Paths
HF_HOME=/data/huggingface                       # Model cache
LOG_LEVEL=INFO                                  # Logging level

# GPU
NVIDIA_VISIBLE_DEVICES=all
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data/huggingface` | `/data/huggingface` | Model cache (persistent) |
| `./data/output` | `/workspace/.../output` | Generated videos |
| `./data/gradio_tmp` | `/workspace/.../gradio_tmp` | Temporary files |

### Useful Commands

```bash
# Start container
docker compose up -d

# View logs
docker logs cogvideo -f

# Check memory usage
docker stats cogvideo

# Restart container
docker restart cogvideo

# Stop container
docker compose down

# Rebuild after changes
docker compose build --no-cache
docker compose up -d

# Execute command inside container
docker exec -it cogvideo bash

# Check GPU usage
docker exec cogvideo nvidia-smi
```

---

## 🔧 Troubleshooting

### High Memory Usage (>20GB)

**Problem:** Container using 20GB+ RAM  
**Cause:** Upscale or interpolation models loaded  
**Solution:** 
```bash
# Restart container to free memory
docker restart cogvideo

# Or avoid using upscale/interpolation features
```

### CUDA Out of Memory

**Problem:** `CUDA out of memory` error  
**Solutions:**
1. Enable CPU offloading in Gradio settings
2. Reduce `num_inference_steps` (try 30-40)
3. Use `CogVideoX-2b` instead of `5b`
4. Disable upscale and interpolation

### Slow Generation

**Problem:** Takes >5 minutes per video  
**Solutions:**
1. Check GPU usage: `nvidia-smi` (should be 90-100%)
2. Reduce steps: 30-40 for testing
3. Enable `enable_sequential_cpu_offload`
4. Check system isn't using swap: `free -h`

### Models Not Downloading

**Problem:** Stuck on "Fetching files"  
**Solutions:**
```bash
# Check disk space
df -h

# Check Hugging Face is accessible
curl -I https://huggingface.co

# Manually download models
huggingface-cli login
huggingface-cli download THUDM/CogVideoX-5b
```

### Port 7860 Already in Use

**Problem:** `bind: address already in use`  
**Solution:**
```bash
# Find and kill process using port 7860
sudo lsof -ti:7860 | xargs kill -9

# Or change port in docker-compose.yaml
ports:
  - "8080:7860"  # Change to 8080
```

---

## 📊 Performance Benchmarks

### RTX 3090 24GB

| Model | Resolution | Frames | Steps | Time | VRAM Peak | Quality |
|-------|------------|--------|-------|------|-----------|---------|
| CogVideoX-2b | 720x480 | 49 | 50 | 45s | 12GB | Good |
| CogVideoX-5b | 720x480 | 49 | 50 | 90s | 18GB | Excellent |
| CogVideoX-5b-I2V | 720x480 | 49 | 50 | 120s | 18GB | Excellent |
| CogVideoX-5b + Upscale | 1440x960 | 49 | 50 | 180s | 22GB | Amazing |

**System Specs:**
- GPU: RTX 3090 24GB
- CPU: Any modern CPU (8+ cores recommended)
- RAM: 32GB (16GB minimum)
- Storage: NVMe SSD (for model loading speed)

---

## 🧪 Testing

### Run All Tests

```bash
cd CogVideo
pytest tests/ -v

# With coverage report
pytest tests/ --cov=inference --cov-report=html

# Run specific test
pytest tests/test_fps_utils.py -v
```

### Test Results

```
✅ 77 tests passing (100%)
⏱️ Test suite: ~30 seconds
📊 Code coverage: 85%+
```

---

## 📚 Documentation

### Quick Links

- 📖 [Full Documentation](CogVideo/docs/_build/html/index.html) (Sphinx)
- 🔧 [Troubleshooting Guide](CogVideo/docs/guides/troubleshooting.md)
- 💾 [VRAM Optimization](CogVideo/docs/guides/vram_optimization.md)
- 🎬 [Motion Presets Guide](CogVideo/docs/guides/motion_presets.md)
- 🚀 [Quickstart Guide](CogVideo/docs/guides/quickstart.md)
- 📝 [AUDIT Report](AUDIT.md) - System optimization analysis

### API Reference

```python
# fps_utils.py
get_correct_fps(model_path: str) -> int

# vram_utils.py
get_vram_usage() -> float
should_offload(model_name: str) -> bool

# motion_presets.py
MOTION_PRESETS = {
    "low_motion": {...},
    "medium_motion": {...},
    "high_motion": {...},
    ...
}

# pipeline_utils.py
load_pipeline_optimized(
    model_path: str,
    motion_preset: str = "balanced",
    enable_tiling: bool = False
) -> CogVideoXPipeline
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

### Development Setup

```bash
# Clone repo
git clone https://github.com/JohnHeberty/CogVideo-Hard-Optimization.git
cd CogVideo-Hard-Optimization

# Install dev dependencies
pip install -r CogVideo/requirements-dev.txt

# Run tests
pytest CogVideo/tests/ -v

# Build docs
cd CogVideo/docs
make html

# Format code
black CogVideo/inference/
ruff check CogVideo/inference/
```

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Model Licenses:**
- CogVideoX models: [THUDM License](CogVideo/MODEL_LICENSE)
- Real-ESRGAN: Apache 2.0
- RIFE: MIT License

---

## 🙏 Acknowledgments

- **THUDM** - Original CogVideoX models
- **HuggingFace** - Diffusers library and model hosting
- **NVIDIA** - CUDA and GPU optimization tools
- **Community** - Testing, feedback, and contributions

---

## 📧 Contact

- **Author:** John Heberty
- **Repository:** [github.com/JohnHeberty/CogVideo-Hard-Optimization](https://github.com/JohnHeberty/CogVideo-Hard-Optimization)
- **Issues:** [GitHub Issues](https://github.com/JohnHeberty/CogVideo-Hard-Optimization/issues)

---

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐

---

<div align="center">
Made with ❤️ for the AI Video Generation Community
</div>
