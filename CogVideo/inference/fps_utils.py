"""
Utilities for FPS (Frames Per Second) calculation and management.

This module provides functions to determine the correct FPS for video export
based on the model used and number of frames generated.
"""

import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# FPS mapping by model family
FPS_MAP: Dict[str, int] = {
    "cogvideox-2b": 8,
    "cogvideox-5b": 8,
    "cogvideox-5b-i2v": 8,
    "cogvideox1.5-5b": 16,
    "cogvideox1.5-5b-i2v": 16,
}

# Expected frame counts and durations by model
FRAME_DURATION_MAP: Dict[str, Dict[str, Any]] = {
    "cogvideox": {
        "base_fps": 8,
        "49_frames": {"duration_seconds": 6, "fps": 8},
    },
    "cogvideox1.5": {
        "base_fps": 16,
        "81_frames": {"duration_seconds": 5, "fps": 16},
        "161_frames": {"duration_seconds": 10, "fps": 16},
    },
}


def get_model_family(model_path: str) -> str:
    """
    Extract model family from model path.
    
    Args:
        model_path: HuggingFace model path (e.g., "THUDM/CogVideoX-5b")
        
    Returns:
        Model family identifier (e.g., "cogvideox", "cogvideox1.5")
    """
    model_name = model_path.split("/")[-1].lower()
    
    if "1.5" in model_name or "1_5" in model_name:
        return "cogvideox1.5"
    else:
        return "cogvideox"


def get_correct_fps(model_path: str, num_frames: Optional[int] = None) -> int:
    """
    Determine the correct FPS for video export based on model and frame count.
    
    CogVideoX models (2B, 5B, 5B-I2V):
        - Generate at 8 fps
        - 49 frames = 6 seconds @ 8fps
        
    CogVideoX1.5 models (5B, 5B-I2V):
        - Generate at 16 fps
        - 81 frames = 5 seconds @ 16fps
        - 161 frames = 10 seconds @ 16fps
    
    Args:
        model_path: HuggingFace model path (e.g., "THUDM/CogVideoX-5b")
        num_frames: Number of frames in the video (optional, for validation)
        
    Returns:
        Correct FPS for video export
        
    Examples:
        >>> get_correct_fps("THUDM/CogVideoX-2b")
        8
        >>> get_correct_fps("THUDM/CogVideoX1.5-5b")
        16
        >>> get_correct_fps("THUDM/CogVideoX-5b-I2V", num_frames=49)
        8
    """
    model_name = model_path.split("/")[-1].lower()
    
    # Get base FPS from model name mapping
    fps = FPS_MAP.get(model_name, None)
    
    if fps is None:
        # Fallback: detect from model family
        family = get_model_family(model_path)
        if family == "cogvideox1.5":
            fps = 16
        else:
            fps = 8
        
        logger.warning(
            f"Model '{model_name}' not in FPS_MAP. "
            f"Using family-based detection: {family} -> {fps}fps"
        )
    
    # Validate frame count if provided
    if num_frames is not None:
        family = get_model_family(model_path)
        expected_info = FRAME_DURATION_MAP.get(family, {})
        
        # Check for known frame counts
        frame_key = f"{num_frames}_frames"
        if frame_key in expected_info:
            expected_fps = expected_info[frame_key]["fps"]
            duration = expected_info[frame_key]["duration_seconds"]
            
            if fps != expected_fps:
                logger.warning(
                    f"FPS mismatch detected. Model: {model_name}, "
                    f"Frames: {num_frames}, Using: {fps}fps (expected: {expected_fps}fps)"
                )
            
            logger.info(
                f"Video: {num_frames} frames @ {fps}fps = {duration}s duration"
            )
        else:
            # Calculate approximate duration
            approx_duration = num_frames / fps
            logger.info(
                f"Video: {num_frames} frames @ {fps}fps ≈ {approx_duration:.1f}s duration"
            )
    
    return fps


def validate_fps_for_model(model_path: str, requested_fps: int, num_frames: Optional[int] = None) -> bool:
    """
    Validate if requested FPS is appropriate for the model.
    
    Args:
        model_path: HuggingFace model path
        requested_fps: FPS requested by user
        num_frames: Number of frames (optional)
        
    Returns:
        True if FPS is appropriate, False otherwise (with warning logged)
    """
    correct_fps = get_correct_fps(model_path, num_frames)
    
    if requested_fps != correct_fps:
        logger.warning(
            f"⚠️  FPS mismatch! You requested {requested_fps}fps, "
            f"but model '{model_path}' generates at {correct_fps}fps. "
            f"Video may appear sped up or slowed down."
        )
        return False
    
    return True


if __name__ == "__main__":
    # Test cases
    logging.basicConfig(level=logging.INFO)
    
    print("Testing FPS detection:")
    print(f"CogVideoX-2b: {get_correct_fps('THUDM/CogVideoX-2b')}fps (expected: 8)")
    print(f"CogVideoX-5b: {get_correct_fps('THUDM/CogVideoX-5b')}fps (expected: 8)")
    print(f"CogVideoX-5b-I2V: {get_correct_fps('THUDM/CogVideoX-5b-I2V')}fps (expected: 8)")
    print(f"CogVideoX1.5-5b: {get_correct_fps('THUDM/CogVideoX1.5-5b')}fps (expected: 16)")
    print(f"CogVideoX1.5-5b-I2V: {get_correct_fps('THUDM/CogVideoX1.5-5b-I2V')}fps (expected: 16)")
    
    print("\nTesting with frame counts:")
    get_correct_fps('THUDM/CogVideoX-5b', num_frames=49)
    get_correct_fps('THUDM/CogVideoX1.5-5b', num_frames=81)
    get_correct_fps('THUDM/CogVideoX1.5-5b', num_frames=161)
    
    print("\nTesting validation:")
    validate_fps_for_model('THUDM/CogVideoX-5b', requested_fps=8)  # Should be OK
    validate_fps_for_model('THUDM/CogVideoX-5b', requested_fps=16)  # Should warn
