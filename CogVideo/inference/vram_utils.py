"""
VRAM utilities for memory management and GPU capability detection.

This module provides functions to check available VRAM, estimate memory requirements,
and determine the best offloading strategy for CogVideoX models.
"""

import logging
import torch
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Estimated VRAM requirements by model (in GB)
# These are approximate values based on bfloat16 precision
MODEL_VRAM_REQUIREMENTS: Dict[str, float] = {
    "cogvideox-2b": 7.5,
    "cogvideox-5b": 17.0,
    "cogvideox-5b-i2v": 17.0,
    "cogvideox1.5-5b": 18.5,
    "cogvideox1.5-5b-i2v": 18.5,
}

# VRAM thresholds for offloading decisions (in GB)
VRAM_THRESHOLDS = {
    "no_offload": 24.0,      # >= 24GB: No offload needed (A100, RTX 4090)
    "model_offload": 16.0,   # >= 16GB: Use model CPU offload (RTX 3090, 4080)
    "sequential_offload": 0, # < 16GB: Use sequential CPU offload (slower but minimal VRAM)
}


def get_gpu_memory_info() -> Tuple[float, float, float]:
    """
    Get current GPU memory usage information.
    
    Returns:
        Tuple of (total_gb, used_gb, available_gb)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU mode.")
        return (0.0, 0.0, 0.0)
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    reserved_memory = torch.cuda.memory_reserved(0)
    
    total_gb = total_memory / (1024**3)
    used_gb = reserved_memory / (1024**3)
    available_gb = (total_memory - reserved_memory) / (1024**3)
    
    return (total_gb, used_gb, available_gb)


def get_model_family(model_path: str) -> str:
    """
    Extract normalized model family from model path.
    
    Args:
        model_path: HuggingFace model path (e.g., "THUDM/CogVideoX-5b")
        
    Returns:
        Normalized model family (e.g., "cogvideox-5b")
    """
    model_name = model_path.split("/")[-1].lower()
    
    # Normalize model name variations
    model_name = model_name.replace("cogvideox1.5", "cogvideox1.5")
    model_name = model_name.replace("cogvideox_", "cogvideox-")
    
    return model_name


def estimate_vram_requirement(model_path: str) -> float:
    """
    Estimate VRAM requirement for a given model.
    
    Args:
        model_path: HuggingFace model path
        
    Returns:
        Estimated VRAM in GB
    """
    model_family = get_model_family(model_path)
    
    # Try exact match first
    if model_family in MODEL_VRAM_REQUIREMENTS:
        return MODEL_VRAM_REQUIREMENTS[model_family]
    
    # Fallback: estimate based on model size indicator
    if "2b" in model_family:
        return MODEL_VRAM_REQUIREMENTS["cogvideox-2b"]
    elif "5b" in model_family:
        if "1.5" in model_family or "1_5" in model_family:
            return MODEL_VRAM_REQUIREMENTS["cogvideox1.5-5b"]
        else:
            return MODEL_VRAM_REQUIREMENTS["cogvideox-5b"]
    
    # Conservative fallback
    logger.warning(f"Unknown model {model_path}, using conservative estimate")
    return 20.0


def check_vram_availability(model_path: str, required_buffer_gb: float = 2.0) -> Tuple[bool, str]:
    """
    Check if there's enough VRAM to load the model.
    
    Args:
        model_path: HuggingFace model path
        required_buffer_gb: Additional buffer to keep free (default: 2GB)
        
    Returns:
        Tuple of (is_sufficient, message)
    """
    total_gb, used_gb, available_gb = get_gpu_memory_info()
    
    if total_gb == 0:
        return (False, "CUDA not available")
    
    estimated_requirement = estimate_vram_requirement(model_path)
    required_vram = estimated_requirement + required_buffer_gb
    
    if available_gb >= required_vram:
        return (True, f"Sufficient VRAM: {available_gb:.1f}GB available, {required_vram:.1f}GB required")
    else:
        return (False, f"Insufficient VRAM: {available_gb:.1f}GB available, {required_vram:.1f}GB required. Consider using CPU offload.")


def get_recommended_offload_strategy(model_path: str) -> str:
    """
    Determine the best CPU offload strategy based on available VRAM.
    
    Strategies:
    - "none": No offload (GPU has enough VRAM)
    - "model": Model CPU offload (balanced speed/memory)
    - "sequential": Sequential CPU offload (slowest but minimal VRAM)
    
    Args:
        model_path: HuggingFace model path
        
    Returns:
        Recommended strategy: "none", "model", or "sequential"
    """
    total_gb, used_gb, available_gb = get_gpu_memory_info()
    
    if total_gb == 0:
        logger.warning("CUDA not available, offloading not applicable")
        return "none"
    
    estimated_requirement = estimate_vram_requirement(model_path)
    
    # Decision logic:
    # 1. If total GPU memory >= 24GB: no offload (A100, 4090)
    # 2. If total GPU memory >= 16GB and available >= requirement: model offload (3090, 4080)
    # 3. Otherwise: sequential offload
    
    if total_gb >= VRAM_THRESHOLDS["no_offload"]:
        logger.info(f"GPU has {total_gb:.1f}GB VRAM (>= 24GB), using direct GPU loading (no offload)")
        return "none"
    elif total_gb >= VRAM_THRESHOLDS["model_offload"]:
        if available_gb >= estimated_requirement * 0.8:  # 80% of requirement
            logger.info(f"GPU has {total_gb:.1f}GB VRAM, using model CPU offload (balanced)")
            return "model"
        else:
            logger.warning(f"GPU has {total_gb:.1f}GB VRAM but only {available_gb:.1f}GB available, using sequential offload")
            return "sequential"
    else:
        logger.info(f"GPU has {total_gb:.1f}GB VRAM (< 16GB), using sequential CPU offload (slow but minimal VRAM)")
        return "sequential"


def apply_offload_strategy(pipe, strategy: str):
    """
    Apply the specified offload strategy to a pipeline.
    
    Args:
        pipe: Diffusers pipeline object
        strategy: "none", "model", or "sequential"
    """
    if strategy == "model":
        logger.info("Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
    elif strategy == "sequential":
        logger.info("Enabling sequential CPU offload...")
        pipe.enable_sequential_cpu_offload()
    elif strategy == "none":
        logger.info("No CPU offload, moving to CUDA...")
        pipe.to("cuda")
    else:
        logger.warning(f"Unknown offload strategy '{strategy}', defaulting to sequential")
        pipe.enable_sequential_cpu_offload()


def log_vram_status(prefix: str = ""):
    """
    Log current VRAM status for debugging.
    
    Args:
        prefix: Optional prefix for the log message
    """
    total_gb, used_gb, available_gb = get_gpu_memory_info()
    
    if total_gb > 0:
        usage_percent = (used_gb / total_gb) * 100
        logger.info(
            f"{prefix}VRAM Status: {used_gb:.2f}GB / {total_gb:.2f}GB used "
            f"({usage_percent:.1f}%), {available_gb:.2f}GB available"
        )
    else:
        logger.info(f"{prefix}CUDA not available")


def configure_vae_tiling(vae, available_vram_gb: Optional[float] = None):
    """
    Configure VAE tiling settings based on available VRAM.
    
    VAE tiling reduces peak memory usage during encoding/decoding by processing
    the video in tiles. This is crucial for generating high-resolution or long videos.
    
    Strategy:
    - >= 20GB VRAM: enable_tiling() only (default tile size)
    - 12-20GB VRAM: enable_tiling() + enable_slicing() 
    - < 12GB VRAM: enable_tiling() + enable_slicing() (already optimal)
    
    Args:
        vae: VAE module from the pipeline
        available_vram_gb: Available VRAM in GB (auto-detected if None)
    """
    if available_vram_gb is None:
        total_gb, _, available_vram_gb = get_gpu_memory_info()
    
    # Always enable slicing and tiling for CogVideoX (recommended in docs)
    vae.enable_slicing()
    vae.enable_tiling()
    
    if available_vram_gb >= 20:
        logger.info("High VRAM (>=20GB): Using standard VAE tiling")
    elif available_vram_gb >= 12:
        logger.info("Medium VRAM (12-20GB): Using VAE tiling + slicing")
    else:
        logger.info("Low VRAM (<12GB): Using aggressive VAE tiling + slicing")
    
    # Note: CogVideoX VAE doesn't expose tile_sample_min_size like SD VAE
    # The enable_tiling() method uses optimal defaults for video generation


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== VRAM Utils Test ===\n")
    
    # Test GPU memory info
    total, used, available = get_gpu_memory_info()
    print(f"GPU Memory: {total:.1f}GB total, {used:.1f}GB used, {available:.1f}GB available")
    
    # Test different models
    test_models = [
        "THUDM/CogVideoX-2b",
        "THUDM/CogVideoX-5b",
        "THUDM/CogVideoX-5b-I2V",
        "THUDM/CogVideoX1.5-5b",
    ]
    
    print("\n=== Model VRAM Requirements ===")
    for model in test_models:
        requirement = estimate_vram_requirement(model)
        print(f"{model}: ~{requirement}GB")
    
    print("\n=== Recommended Offload Strategies ===")
    for model in test_models:
        strategy = get_recommended_offload_strategy(model)
        print(f"{model}: {strategy}")
    
    print("\n=== VRAM Availability Check ===")
    for model in test_models:
        is_ok, msg = check_vram_availability(model)
        status = "✓" if is_ok else "✗"
        print(f"{status} {model}: {msg}")
