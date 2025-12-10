"""
Motion quality presets for CogVideoX video generation.

This module provides preset configurations optimized for different types of motion
and use cases. These presets tune parameters like guidance_scale, num_inference_steps,
and other settings to achieve better results for specific scenarios.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MotionPreset:
    """Configuration preset for video generation."""
    name: str
    description: str
    guidance_scale: float
    num_inference_steps: int
    use_dynamic_cfg: bool
    recommended_for: List[str]
    notes: str = ""


# Preset definitions based on CogVideoX best practices
MOTION_PRESETS: Dict[str, MotionPreset] = {
    "balanced": MotionPreset(
        name="Balanced",
        description="Balanced quality and speed (default)",
        guidance_scale=6.0,
        num_inference_steps=50,
        use_dynamic_cfg=True,
        recommended_for=["general use", "mixed content", "testing"],
        notes="Good starting point for most prompts. 6-8 seconds generation time."
    ),
    
    "fast": MotionPreset(
        name="Fast",
        description="Faster generation with acceptable quality",
        guidance_scale=5.0,
        num_inference_steps=30,
        use_dynamic_cfg=True,
        recommended_for=["previews", "iterations", "low-motion scenes"],
        notes="~40% faster than balanced. Best for static or slow-moving scenes."
    ),
    
    "quality": MotionPreset(
        name="Quality",
        description="Higher quality, slower generation",
        guidance_scale=7.0,
        num_inference_steps=75,
        use_dynamic_cfg=True,
        recommended_for=["final renders", "complex motion", "detailed scenes"],
        notes="~50% slower than balanced. Better temporal consistency and detail."
    ),
    
    "high_motion": MotionPreset(
        name="High Motion",
        description="Optimized for fast-moving subjects",
        guidance_scale=6.5,
        num_inference_steps=60,
        use_dynamic_cfg=True,
        recommended_for=["sports", "action", "running", "dancing", "fast camera moves"],
        notes="Higher guidance helps maintain subject coherence during rapid motion. "
              "Use descriptive motion keywords: 'sprinting', 'leaping', 'rapid'."
    ),
    
    "subtle": MotionPreset(
        name="Subtle Motion",
        description="For gentle, natural movements",
        guidance_scale=5.5,
        num_inference_steps=55,
        use_dynamic_cfg=True,
        recommended_for=["portraits", "nature", "slow pans", "ambient scenes"],
        notes="Lower guidance reduces over-animation. Good for: 'gentle breeze', "
              "'slowly walking', 'calm water', 'subtle facial expressions'."
    ),
}


# Recommended prompts for each preset (for user guidance)
PROMPT_EXAMPLES: Dict[str, List[str]] = {
    "balanced": [
        "A golden retriever running through a field",
        "A person walking down a city street",
        "Ocean waves on a beach at sunset",
    ],
    
    "fast": [
        "A cat sitting on a windowsill",
        "A flower blooming in time-lapse",
        "Clouds drifting across the sky",
    ],
    
    "quality": [
        "A ballet dancer performing a pirouette on stage",
        "A chef preparing an elaborate dish in a kitchen",
        "A waterfall cascading into a crystal-clear pool",
    ],
    
    "high_motion": [
        "A golden retriever sprinting playfully across a meadow, ears flapping",
        "A basketball player dunking in slow motion",
        "A Formula 1 car racing around a track at high speed",
        "A parkour athlete leaping between rooftops",
    ],
    
    "subtle": [
        "A woman with long hair gently swaying in the breeze",
        "Soft candlelight flickering on a table",
        "A leaf slowly falling from a tree",
        "A person's eyes slowly opening in the morning",
    ],
}


def get_preset(preset_name: str) -> MotionPreset:
    """
    Get motion preset by name.
    
    Args:
        preset_name: Name of the preset ("balanced", "fast", "quality", "high_motion", "subtle")
        
    Returns:
        MotionPreset object
        
    Raises:
        ValueError: If preset_name is not found
    """
    preset_name = preset_name.lower()
    
    if preset_name not in MOTION_PRESETS:
        available = ", ".join(MOTION_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    
    return MOTION_PRESETS[preset_name]


def list_presets() -> List[str]:
    """
    Get list of available preset names.
    
    Returns:
        List of preset names
    """
    return list(MOTION_PRESETS.keys())


def get_preset_info(preset_name: str) -> str:
    """
    Get human-readable information about a preset.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Formatted string with preset details
    """
    preset = get_preset(preset_name)
    
    info = f"""
Preset: {preset.name}
Description: {preset.description}
Settings:
  - Guidance Scale: {preset.guidance_scale}
  - Inference Steps: {preset.num_inference_steps}
  - Dynamic CFG: {preset.use_dynamic_cfg}

Recommended For:
  {', '.join(preset.recommended_for)}

{preset.notes}
""".strip()
    
    return info


def apply_preset_to_pipeline_args(preset_name: str, base_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a motion preset to pipeline generation arguments.
    
    Args:
        preset_name: Name of the preset to apply
        base_args: Base arguments dict (will not be modified)
        
    Returns:
        New dict with preset values applied
        
    Example:
        >>> args = {"prompt": "A dog running", "num_frames": 49}
        >>> args_with_preset = apply_preset_to_pipeline_args("high_motion", args)
        >>> args_with_preset["guidance_scale"]
        6.5
    """
    preset = get_preset(preset_name)
    
    # Create new dict to avoid modifying input
    result = base_args.copy()
    
    # Apply preset values (only if not already specified)
    if "guidance_scale" not in result:
        result["guidance_scale"] = preset.guidance_scale
    
    if "num_inference_steps" not in result:
        result["num_inference_steps"] = preset.num_inference_steps
    
    if "use_dynamic_cfg" not in result:
        result["use_dynamic_cfg"] = preset.use_dynamic_cfg
    
    logger.info(f"Applied '{preset.name}' preset to pipeline arguments")
    return result


def get_prompt_examples(preset_name: str) -> List[str]:
    """
    Get example prompts for a given preset.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        List of example prompts
    """
    preset_name = preset_name.lower()
    
    if preset_name not in PROMPT_EXAMPLES:
        logger.warning(f"No examples for preset '{preset_name}'")
        return []
    
    return PROMPT_EXAMPLES[preset_name]


def print_all_presets():
    """Print information about all available presets."""
    print("\n" + "="*80)
    print("CogVideoX Motion Presets")
    print("="*80 + "\n")
    
    for preset_name in MOTION_PRESETS.keys():
        preset = MOTION_PRESETS[preset_name]
        print(f"[{preset_name.upper()}]")
        print(f"  {preset.description}")
        print(f"  Guidance: {preset.guidance_scale} | Steps: {preset.num_inference_steps}")
        print(f"  Best for: {', '.join(preset.recommended_for)}")
        
        if preset.notes:
            print(f"  Note: {preset.notes[:80]}...")
        
        print()


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Motion Presets Test ===\n")
    
    # Test listing presets
    presets = list_presets()
    print(f"Available presets: {presets}\n")
    
    # Test getting preset
    high_motion = get_preset("high_motion")
    print(f"High Motion preset:")
    print(f"  Guidance: {high_motion.guidance_scale}")
    print(f"  Steps: {high_motion.num_inference_steps}\n")
    
    # Test applying preset
    base_args = {"prompt": "A dog running", "num_frames": 49}
    args_with_preset = apply_preset_to_pipeline_args("high_motion", base_args)
    print(f"Args with preset: {args_with_preset}\n")
    
    # Test getting examples
    examples = get_prompt_examples("high_motion")
    print(f"Example prompts for high_motion:")
    for ex in examples:
        print(f"  - {ex}")
    
    print("\n" + "="*80)
    print_all_presets()
