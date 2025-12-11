"""
Test suite for motion_presets module.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))

from motion_presets import (
    MotionPreset,
    MOTION_PRESETS,
    get_preset,
    list_presets,
    apply_preset_to_pipeline_args
)


class TestMotionPreset:
    """Tests for MotionPreset dataclass."""
    
    def test_preset_creation(self):
        """Should create preset with required fields."""
        preset = MotionPreset(
            name="test",
            description="Test preset",
            guidance_scale=6.0,
            num_inference_steps=50,
            use_dynamic_cfg=True,
            recommended_for=["testing"]
        )
        assert preset.name == "test"
        assert preset.guidance_scale == 6.0
        assert preset.num_inference_steps == 50
    
    def test_preset_has_all_fields(self):
        """Should have all expected fields."""
        preset = MotionPreset(
            name="test",
            description="Test",
            guidance_scale=6.0,
            num_inference_steps=50,
            use_dynamic_cfg=True,
            recommended_for=["testing"]
        )
        assert hasattr(preset, "use_dynamic_cfg")
        assert hasattr(preset, "recommended_for")


class TestMotionPresetsConstant:
    """Tests for MOTION_PRESETS constant."""
    
    def test_motion_presets_exists(self):
        """MOTION_PRESETS should be defined."""
        assert MOTION_PRESETS is not None
        assert isinstance(MOTION_PRESETS, dict)
    
    def test_has_required_presets(self):
        """Should have all 5 required presets."""
        required = ["balanced", "fast", "quality", "high_motion", "subtle"]
        for preset_name in required:
            assert preset_name in MOTION_PRESETS, f"Missing preset: {preset_name}"
    
    def test_preset_count(self):
        """Should have exactly 5 presets."""
        assert len(MOTION_PRESETS) == 5
    
    def test_all_presets_valid(self):
        """All presets should be MotionPreset instances."""
        for name, preset in MOTION_PRESETS.items():
            assert isinstance(preset, MotionPreset)
            # Preset.name is display name (capitalized), key is lowercase
            assert preset.name.lower().replace(" ", "_") == name or name in preset.name.lower()


class TestGetPreset:
    """Tests for get_preset function."""
    
    def test_get_balanced(self):
        """Should get balanced preset."""
        preset = get_preset("balanced")
        assert "balanced" in preset.name.lower()
        assert preset.guidance_scale == 6.0
        assert preset.num_inference_steps == 50
    
    def test_get_fast(self):
        """Should get fast preset."""
        preset = get_preset("fast")
        assert "fast" in preset.name.lower()
        assert preset.guidance_scale == 5.0
        assert preset.num_inference_steps == 30
    
    def test_get_quality(self):
        """Should get quality preset."""
        preset = get_preset("quality")
        assert "quality" in preset.name.lower()
        assert preset.guidance_scale == 7.0
        assert preset.num_inference_steps == 75
    
    def test_get_high_motion(self):
        """Should get high_motion preset (golden retriever fix)."""
        preset = get_preset("high_motion")
        assert "motion" in preset.name.lower()
        assert preset.guidance_scale == 6.5
        assert preset.num_inference_steps == 60
    
    def test_get_subtle(self):
        """Should get subtle preset."""
        preset = get_preset("subtle")
        assert "subtle" in preset.name.lower()
        assert preset.guidance_scale == 5.5
        assert preset.num_inference_steps == 55
    
    def test_get_nonexistent_raises_error(self):
        """Non-existent preset should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_preset("nonexistent")
        assert "Unknown preset" in str(exc_info.value)
    
    def test_get_case_insensitive(self):
        """Should handle different cases (if implemented)."""
        # Try uppercase
        preset = get_preset("BALANCED")
        # Should either work or fallback to balanced
        assert preset is not None


class TestListPresets:
    """Tests for list_presets function."""
    
    def test_returns_list(self):
        """Should return list of strings."""
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) == 5
    
    def test_list_contains_strings(self):
        """Each item should be a string."""
        presets = list_presets()
        for item in presets:
            assert isinstance(item, str)
    
    def test_all_presets_listed(self):
        """All 5 presets should be in list."""
        presets = list_presets()
        assert "balanced" in presets
        assert "fast" in presets
        assert "quality" in presets
        assert "high_motion" in presets
        assert "subtle" in presets


class TestApplyPresetToPipelineArgs:
    """Tests for apply_preset_to_pipeline_args function."""
    
    def test_apply_balanced(self):
        """Should apply balanced preset to args."""
        args = {"prompt": "test"}
        result = apply_preset_to_pipeline_args("balanced", args)
        
        assert result["prompt"] == "test"  # Original preserved
        assert result["guidance_scale"] == 6.0
        assert result["num_inference_steps"] == 50
    
    def test_apply_high_motion(self):
        """Should apply high_motion preset to args."""
        args = {"prompt": "golden retriever", "num_frames": 49}
        result = apply_preset_to_pipeline_args("high_motion", args)
        
        assert result["prompt"] == "golden retriever"
        assert result["num_frames"] == 49
        assert result["guidance_scale"] == 6.5
        assert result["num_inference_steps"] == 60
    
    def test_preserves_existing_args(self):
        """Should preserve all existing args."""
        args = {
            "prompt": "test",
            "negative_prompt": "blurry",
            "num_frames": 49,
            "seed": 42
        }
        result = apply_preset_to_pipeline_args("fast", args)
        
        assert result["prompt"] == "test"
        assert result["negative_prompt"] == "blurry"
        assert result["num_frames"] == 49
        assert result["seed"] == 42
    
    def test_does_not_modify_original(self):
        """Should not modify original args dict."""
        original = {"prompt": "test"}
        result = apply_preset_to_pipeline_args("quality", original)
        
        # Original should be unchanged
        assert "guidance_scale" not in original
        # Result should have preset values
        assert "guidance_scale" in result
    
    def test_empty_args_dict(self):
        """Should work with empty args."""
        result = apply_preset_to_pipeline_args("balanced", {})
        assert "guidance_scale" in result
        assert "num_inference_steps" in result


class TestPresetValues:
    """Tests for specific preset values (golden retriever fix validation)."""
    
    def test_balanced_preset_values(self):
        """Balanced preset should have documented values."""
        preset = get_preset("balanced")
        assert preset.guidance_scale == 6.0
        assert preset.num_inference_steps == 50
    
    def test_fast_preset_faster_than_balanced(self):
        """Fast preset should have fewer steps."""
        fast = get_preset("fast")
        balanced = get_preset("balanced")
        assert fast.num_inference_steps < balanced.num_inference_steps
    
    def test_quality_preset_slower_than_balanced(self):
        """Quality preset should have more steps."""
        quality = get_preset("quality")
        balanced = get_preset("balanced")
        assert quality.num_inference_steps > balanced.num_inference_steps
    
    def test_high_motion_guidance_higher_than_balanced(self):
        """High motion should have higher guidance (fixes golden retriever)."""
        high_motion = get_preset("high_motion")
        balanced = get_preset("balanced")
        assert high_motion.guidance_scale > balanced.guidance_scale
    
    def test_subtle_lower_guidance(self):
        """Subtle preset should have lower guidance."""
        subtle = get_preset("subtle")
        balanced = get_preset("balanced")
        assert subtle.guidance_scale < balanced.guidance_scale


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_none_preset_name(self):
        """None preset name should raise error."""
        with pytest.raises((ValueError, AttributeError)):
            get_preset(None)
    
    def test_empty_string_preset(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            get_preset("")
    
    def test_apply_with_none_args(self):
        """Apply preset with None args should raise error."""
        with pytest.raises(AttributeError):
            apply_preset_to_pipeline_args("balanced", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
