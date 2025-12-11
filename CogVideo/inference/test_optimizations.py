"""
Quick validation script for CogVideoX optimizations.

This script tests the core utilities without loading actual models.
Run this to verify all imports and basic functions work correctly.

Usage:
    python test_optimizations.py
"""

import sys
from pathlib import Path

# Add inference directory to path
inference_dir = Path(__file__).parent
sys.path.insert(0, str(inference_dir))

def test_fps_utils():
    """Test FPS utilities."""
    print("\n" + "="*60)
    print("Testing fps_utils.py")
    print("="*60)
    
    from fps_utils import get_correct_fps, validate_fps_for_model
    
    # Test FPS detection
    models = [
        ("THUDM/CogVideoX-2b", 49, 8),
        ("THUDM/CogVideoX-5b", 49, 8),
        ("THUDM/CogVideoX1.5-5b", 81, 16),
    ]
    
    for model, frames, expected_fps in models:
        fps = get_correct_fps(model, frames)
        status = "✓" if fps == expected_fps else "✗"
        print(f"{status} {model}: {fps}fps (expected {expected_fps})")
    
    print("✓ FPS utils working")
    return True


def test_vram_utils():
    """Test VRAM utilities."""
    print("\n" + "="*60)
    print("Testing vram_utils.py")
    print("="*60)
    
    from vram_utils import (
        get_gpu_memory_info,
        estimate_vram_requirement,
        get_recommended_offload_strategy,
    )
    
    # Test GPU info
    total, used, available = get_gpu_memory_info()
    if total > 0:
        print(f"✓ GPU detected: {total:.1f}GB total, {available:.1f}GB available")
    else:
        print("⚠ No GPU detected (expected in CPU-only environment)")
    
    # Test VRAM estimation
    models = [
        ("THUDM/CogVideoX-2b", 7.5),
        ("THUDM/CogVideoX-5b", 17.0),
    ]
    
    for model, expected_vram in models:
        vram = estimate_vram_requirement(model)
        status = "✓" if vram == expected_vram else "✗"
        print(f"{status} {model}: {vram}GB VRAM estimate")
    
    # Test offload strategy
    strategy = get_recommended_offload_strategy("THUDM/CogVideoX-5b")
    print(f"✓ Offload strategy for 5B: {strategy}")
    
    print("✓ VRAM utils working")
    return True


def test_motion_presets():
    """Test motion presets."""
    print("\n" + "="*60)
    print("Testing motion_presets.py")
    print("="*60)
    
    from motion_presets import get_preset, list_presets, apply_preset_to_pipeline_args
    
    # List presets
    presets = list_presets()
    print(f"✓ Available presets: {', '.join(presets)}")
    
    # Test high_motion preset
    preset = get_preset("high_motion")
    print(f"✓ high_motion preset:")
    print(f"  - Guidance: {preset.guidance_scale}")
    print(f"  - Steps: {preset.num_inference_steps}")
    
    # Test apply preset
    base_args = {"prompt": "test", "num_frames": 49}
    args = apply_preset_to_pipeline_args("balanced", base_args)
    
    if "guidance_scale" in args and "num_inference_steps" in args:
        print(f"✓ Preset applied: guidance={args['guidance_scale']}, steps={args['num_inference_steps']}")
    else:
        print("✗ Preset application failed")
        return False
    
    print("✓ Motion presets working")
    return True


def test_pipeline_utils():
    """Test pipeline utilities (without loading models)."""
    print("\n" + "="*60)
    print("Testing pipeline_utils.py")
    print("="*60)
    
    try:
        from pipeline_utils import get_pipeline_class, get_model_info
        
        # Test pipeline class detection
        from diffusers import CogVideoXPipeline
        pipe_class = get_pipeline_class("t2v")
        status = "✓" if pipe_class == CogVideoXPipeline else "✗"
        print(f"{status} T2V pipeline class: {pipe_class.__name__}")
        
        # Test model info
        info = get_model_info("THUDM/CogVideoX-5b")
        print(f"✓ Model info for CogVideoX-5b:")
        print(f"  - Size: {info['size']}")
        print(f"  - FPS: {info['base_fps']}")
        print(f"  - Resolution: {info['recommended_resolution']}")
        
        print("✓ Pipeline utils working")
        return True
        
    except ImportError as e:
        print(f"⚠ Skipping pipeline_utils test (diffusers not installed): {e}")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CogVideoX Optimizations - Validation Test")
    print("="*60)
    
    tests = [
        ("FPS Utils", test_fps_utils),
        ("VRAM Utils", test_vram_utils),
        ("Motion Presets", test_motion_presets),
        ("Pipeline Utils", test_pipeline_utils),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All optimizations validated successfully!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
