FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface \
    HF_HUB_CACHE=/data/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sistema + ffmpeg + curl (required for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copia a pasta CogVideo local (com todas as otimizações implementadas)
# ao invés de clonar do GitHub
COPY CogVideo /workspace/CogVideo

WORKDIR /workspace/CogVideo

# Cria venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# PyTorch 2.5.1+ CUDA 12.1 (cu121) - Versão mais recente disponível
# Note: PyTorch 2.8.0 ainda não existe, usando 2.5.1
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 torchvision torchaudio

# Dependências da demo composta (inclui diffusers, gradio, accelerate, etc.)
RUN pip install -r inference/gradio_composite_demo/requirements.txt

# Instala imageio-ffmpeg para H.264 export optimization
RUN pip install imageio imageio-ffmpeg

WORKDIR /workspace/CogVideo/inference/gradio_composite_demo

EXPOSE 7860

# Usa o app.py otimizado (já tem lazy load + component sharing + todas otimizações)
# ao invés do lite_webui.py
CMD ["python", "app.py"]
