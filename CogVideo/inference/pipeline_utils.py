"""
Pipeline utilities for CogVideoX model loading and configuration.

This module centralizes pipeline loading logic with automatic:
- Model type detection (T2V, I2V, V2V)
- VRAM-based offload strategy selection
- VAE tiling configuration
- Scheduler configuration
- LoRA loading

Usage:
    from pipeline_utils import load_pipeline
    
    pipe = load_pipeline(
        model_path="THUDM/CogVideoX-5b",
        pipeline_type="t2v",
        dtype=torch.bfloat16,
        lora_path=None
    )
"""

import logging
import torch
from typing import Optional, Literal
from diffusers import (
    CogVideoXPipeline,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
)

from vram_utils import (
    get_recommended_offload_strategy,
    apply_offload_strategy,
    configure_vae_tiling,
    log_vram_status,
)

logger = logging.getLogger(__name__)

# Type alias for pipeline types
PipelineType = Literal["t2v", "i2v", "v2v"]


def get_pipeline_class(pipeline_type: PipelineType):
    """
    Get the appropriate pipeline class for the given type.
    
    Args:
        pipeline_type: Type of pipeline ("t2v", "i2v", "v2v")
        
    Returns:
        Pipeline class
    """
    pipeline_classes = {
        "t2v": CogVideoXPipeline,
        "i2v": CogVideoXImageToVideoPipeline,
        "v2v": CogVideoXVideoToVideoPipeline,
    }
    
    if pipeline_type not in pipeline_classes:
        raise ValueError(
            f"Unknown pipeline type '{pipeline_type}'. "
            f"Must be one of: {list(pipeline_classes.keys())}"
        )
    
    return pipeline_classes[pipeline_type]


def configure_scheduler(
    pipe,
    scheduler_type: Literal["dpm", "ddim"] = "dpm",
    timestep_spacing: str = "trailing"
):
    """
    Configure the scheduler for a pipeline.
    
    Args:
        pipe: Pipeline object
        scheduler_type: Type of scheduler ("dpm" or "ddim")
        timestep_spacing: Timestep spacing strategy
        
    Note:
        - DPM recommended for CogVideoX-5B and CogVideoX1.5
        - DDIM recommended for CogVideoX-2B
    """
    if scheduler_type == "ddim":
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing=timestep_spacing
        )
        logger.info("Configured DDIM scheduler")
    else:
        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing=timestep_spacing
        )
        logger.info("Configured DPM scheduler")


def load_pipeline(
    model_path: str,
    pipeline_type: PipelineType = "t2v",
    dtype: torch.dtype = torch.bfloat16,
    lora_path: Optional[str] = None,
    lora_rank: int = 128,
    scheduler_type: Literal["dpm", "ddim"] = "dpm",
    enable_offload: bool = True,
    offload_strategy: Optional[str] = None,
) -> CogVideoXPipeline:
    """
    Load and configure a CogVideoX pipeline with optimal settings.
    
    This function handles:
    1. Pipeline instantiation (T2V, I2V, or V2V)
    2. LoRA loading (if specified)
    3. Scheduler configuration
    4. VRAM-based offload strategy
    5. VAE tiling optimization
    
    Args:
        model_path: HuggingFace model path (e.g., "THUDM/CogVideoX-5b")
        pipeline_type: Type of pipeline ("t2v", "i2v", "v2v")
        dtype: Data type for computation (torch.bfloat16 or torch.float16)
        lora_path: Optional path to LoRA weights
        lora_rank: Rank for LoRA weights (default: 128)
        scheduler_type: Scheduler type ("dpm" or "ddim")
        enable_offload: Whether to enable CPU offload (default: True)
        offload_strategy: Manual offload strategy ("none", "model", "sequential").
                         If None, automatically determined based on VRAM.
        
    Returns:
        Configured pipeline ready for inference
        
    Example:
        >>> pipe = load_pipeline(
        ...     model_path="THUDM/CogVideoX-5b",
        ...     pipeline_type="t2v",
        ...     dtype=torch.bfloat16
        ... )
        >>> video = pipe(prompt="A dog running", num_frames=49).frames[0]
    """
    logger.info(f"Loading {pipeline_type.upper()} pipeline from {model_path}")
    log_vram_status("[Before Pipeline Load] ")
    
    # 1. Load base pipeline
    pipeline_class = get_pipeline_class(pipeline_type)
    pipe = pipeline_class.from_pretrained(model_path, torch_dtype=dtype)
    
    # 2. Load LoRA if specified
    if lora_path:
        logger.info(f"Loading LoRA weights from {lora_path}")
        pipe.load_lora_weights(
            lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="lora_adapter"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)
        logger.info(f"LoRA loaded and fused (rank={lora_rank})")
    
    # 3. Configure scheduler
    configure_scheduler(pipe, scheduler_type=scheduler_type)
    
    # 4. Apply offload strategy
    if enable_offload:
        if offload_strategy is None:
            # Auto-detect based on VRAM
            offload_strategy = get_recommended_offload_strategy(model_path)
        
        apply_offload_strategy(pipe, offload_strategy)
    else:
        logger.info("Offload disabled, moving entire pipeline to CUDA")
        pipe.to("cuda")
    
    # 5. Configure VAE tiling
    configure_vae_tiling(pipe.vae)
    
    log_vram_status("[After Pipeline Load] ")
    logger.info(f"Pipeline loaded successfully ({pipeline_type.upper()})")
    
    return pipe


def load_shared_pipeline(
    model_path: str,
    target_pipeline_type: PipelineType,
    base_pipeline: Optional[CogVideoXPipeline] = None,
    dtype: torch.dtype = torch.bfloat16,
    scheduler_type: Literal["dpm", "ddim"] = "dpm",
    enable_offload: bool = True,
) -> CogVideoXPipeline:
    """
    Load a pipeline that shares components with an existing base pipeline.
    
    This is useful for V2V pipelines that can share transformer/VAE with T2V,
    saving ~10GB VRAM.
    
    Args:
        model_path: HuggingFace model path
        target_pipeline_type: Type of pipeline to load ("v2v", "i2v", etc.)
        base_pipeline: Existing pipeline to share components from (typically T2V)
        dtype: Data type for computation
        scheduler_type: Scheduler type
        enable_offload: Whether to enable CPU offload
        
    Returns:
        New pipeline sharing components with base_pipeline
        
    Example:
        >>> t2v_pipe = load_pipeline("THUDM/CogVideoX-5b", "t2v")
        >>> v2v_pipe = load_shared_pipeline(
        ...     "THUDM/CogVideoX-5b", "v2v", base_pipeline=t2v_pipe
        ... )
    """
    if base_pipeline is None:
        logger.warning("No base pipeline provided, loading standalone pipeline")
        return load_pipeline(
            model_path=model_path,
            pipeline_type=target_pipeline_type,
            dtype=dtype,
            scheduler_type=scheduler_type,
            enable_offload=enable_offload,
        )
    
    logger.info(f"Loading {target_pipeline_type.upper()} with shared components")
    
    pipeline_class = get_pipeline_class(target_pipeline_type)
    
    # Share transformer, VAE, scheduler, text encoder, and tokenizer
    pipe = pipeline_class.from_pretrained(
        model_path,
        transformer=base_pipeline.transformer,
        vae=base_pipeline.vae,
        scheduler=base_pipeline.scheduler,
        tokenizer=base_pipeline.tokenizer,
        text_encoder=base_pipeline.text_encoder,
        torch_dtype=dtype,
    )
    
    if enable_offload:
        offload_strategy = get_recommended_offload_strategy(model_path)
        apply_offload_strategy(pipe, offload_strategy)
    else:
        pipe.to("cuda")
    
    logger.info(f"Shared pipeline loaded ({target_pipeline_type.upper()})")
    return pipe


def get_model_info(model_path: str) -> dict:
    """
    Get information about a model.
    
    Args:
        model_path: HuggingFace model path
        
    Returns:
        Dict with model information
    """
    model_name = model_path.split("/")[-1].lower()
    
    info = {
        "model_path": model_path,
        "model_name": model_name,
        "is_i2v": "i2v" in model_name,
        "is_1_5": "1.5" in model_name or "1_5" in model_name,
        "base_fps": 16 if ("1.5" in model_name or "1_5" in model_name) else 8,
    }
    
    # Determine model size
    if "2b" in model_name:
        info["size"] = "2B"
        info["recommended_scheduler"] = "ddim"
    elif "5b" in model_name:
        info["size"] = "5B"
        info["recommended_scheduler"] = "dpm"
    else:
        info["size"] = "unknown"
        info["recommended_scheduler"] = "dpm"
    
    # Recommended resolution
    if info["is_1_5"]:
        info["recommended_resolution"] = (768, 1360)  # height, width
    else:
        info["recommended_resolution"] = (480, 720)
    
    # Recommended frame counts
    if info["is_1_5"]:
        info["recommended_frames"] = [81, 161]  # 5s, 10s @ 16fps
    else:
        info["recommended_frames"] = [49]  # 6s @ 8fps
    
    return info


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Pipeline Utils Test ===\n")
    
    # Test model info
    models = [
        "THUDM/CogVideoX-2b",
        "THUDM/CogVideoX-5b",
        "THUDM/CogVideoX-5b-I2V",
        "THUDM/CogVideoX1.5-5b",
    ]
    
    for model in models:
        info = get_model_info(model)
        print(f"{model}:")
        print(f"  Size: {info['size']}")
        print(f"  FPS: {info['base_fps']}")
        print(f"  Resolution: {info['recommended_resolution']}")
        print(f"  Scheduler: {info['recommended_scheduler']}")
        print()
