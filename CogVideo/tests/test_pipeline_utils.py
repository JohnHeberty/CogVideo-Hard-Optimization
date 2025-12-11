"""
Test suite for pipeline_utils module.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))

from pipeline_utils import (
    get_pipeline_class,
    get_model_info,
    validate_model_path
)


class TestGetPipelineClass:
    """Tests for get_pipeline_class function."""
    
    def test_get_t2v_class(self):
        """Should return CogVideoXPipeline for T2V."""
        cls = get_pipeline_class("t2v")
        assert cls is not None
        assert "CogVideoX" in cls.__name__
    
    def test_get_i2v_class(self):
        """Should return CogVideoXImageToVideoPipeline for I2V."""
        cls = get_pipeline_class("i2v")
        assert cls is not None
        assert "ImageToVideo" in cls.__name__ or "I2V" in cls.__name__
    
    def test_get_v2v_class(self):
        """Should return CogVideoXVideoToVideoPipeline for V2V."""
        cls = get_pipeline_class("v2v")
        assert cls is not None
        assert "VideoToVideo" in cls.__name__ or "V2V" in cls.__name__
    
    def test_invalid_type_raises_error(self):
        """Invalid pipeline type should raise ValueError."""
        with pytest.raises(ValueError):
            get_pipeline_class("invalid_type")
    
    def test_case_insensitive(self):
        """Should handle different cases."""
        cls_lower = get_pipeline_class("t2v")
        cls_upper = get_pipeline_class("T2V")
        assert cls_lower == cls_upper


class TestGetModelInfo:
    """Tests for get_model_info function."""
    
    def test_cogvideox_2b_info(self):
        """Should return correct info for CogVideoX-2b."""
        info = get_model_info("THUDM/CogVideoX-2b")
        assert info["type"] == "t2v"
        assert info["default_fps"] == 8
        assert info["default_frames"] == 49
        assert info["vram_gb"] == 8.0
    
    def test_cogvideox_5b_info(self):
        """Should return correct info for CogVideoX-5b."""
        info = get_model_info("THUDM/CogVideoX-5b")
        assert info["type"] == "t2v"
        assert info["default_fps"] == 8
        assert info["default_frames"] == 49
        assert info["vram_gb"] == 17.0
    
    def test_cogvideox_5b_i2v_info(self):
        """Should return correct info for CogVideoX-5b-I2V."""
        info = get_model_info("THUDM/CogVideoX-5b-I2V")
        assert info["type"] == "i2v"
        assert info["default_fps"] == 8
        assert info["default_frames"] == 49
        assert info["vram_gb"] == 18.0
    
    def test_cogvideox1_5_5b_info(self):
        """Should return correct info for CogVideoX1.5-5b."""
        info = get_model_info("THUDM/CogVideoX1.5-5b")
        assert info["type"] == "t2v"
        assert info["default_fps"] == 16
        assert info["default_frames"] == 81
        assert info["vram_gb"] == 20.0
    
    def test_cogvideox1_5_i2v_info(self):
        """Should return correct info for CogVideoX1.5-5b-I2V."""
        info = get_model_info("THUDM/CogVideoX1.5-5b-I2V")
        assert info["type"] == "i2v"
        assert info["default_fps"] == 16
        assert info["default_frames"] == 81
    
    def test_local_model_path(self):
        """Should work with local model paths."""
        info = get_model_info("./models/CogVideoX-5b")
        assert info["type"] == "t2v"
        assert info["default_fps"] == 8
    
    def test_model_info_structure(self):
        """Model info should have required keys."""
        info = get_model_info("THUDM/CogVideoX-5b")
        required_keys = ["type", "default_fps", "default_frames", "vram_gb"]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"


class TestValidateModelPath:
    """Tests for validate_model_path function."""
    
    def test_valid_cogvideox_2b(self):
        """CogVideoX-2b should validate."""
        try:
            validate_model_path("THUDM/CogVideoX-2b")
        except ValueError:
            pytest.fail("Valid model path should not raise ValueError")
    
    def test_valid_cogvideox_5b(self):
        """CogVideoX-5b should validate."""
        try:
            validate_model_path("THUDM/CogVideoX-5b")
        except ValueError:
            pytest.fail("Valid model path should not raise ValueError")
    
    def test_valid_cogvideox_i2v(self):
        """CogVideoX-5b-I2V should validate."""
        try:
            validate_model_path("THUDM/CogVideoX-5b-I2V")
        except ValueError:
            pytest.fail("Valid model path should not raise ValueError")
    
    def test_valid_cogvideox1_5(self):
        """CogVideoX1.5 should validate."""
        try:
            validate_model_path("THUDM/CogVideoX1.5-5b")
        except ValueError:
            pytest.fail("Valid model path should not raise ValueError")
    
    def test_invalid_model_raises_error(self):
        """Invalid model path should raise ValueError."""
        with pytest.raises(ValueError):
            validate_model_path("invalid/model")
    
    def test_empty_path_raises_error(self):
        """Empty path should raise ValueError."""
        with pytest.raises(ValueError):
            validate_model_path("")
    
    def test_local_path_validation(self):
        """Local paths with correct model names should validate."""
        try:
            validate_model_path("./models/CogVideoX-5b")
        except ValueError:
            pytest.fail("Local path with valid model name should validate")


class TestModelInfoConsistency:
    """Tests for consistency between different info sources."""
    
    def test_fps_matches_fps_utils(self):
        """FPS from model_info should match fps_utils."""
        from fps_utils import get_correct_fps
        
        models_to_test = [
            ("THUDM/CogVideoX-2b", 49),
            ("THUDM/CogVideoX-5b", 49),
            ("THUDM/CogVideoX1.5-5b", 81)
        ]
        
        for model_path, num_frames in models_to_test:
            info = get_model_info(model_path)
            fps_from_utils = get_correct_fps(model_path, num_frames)
            
            assert info["default_fps"] == fps_from_utils, \
                f"FPS mismatch for {model_path}"
    
    def test_vram_matches_vram_utils(self):
        """VRAM from model_info should match vram_utils."""
        from vram_utils import estimate_vram_requirement
        
        models_to_test = [
            "THUDM/CogVideoX-2b",
            "THUDM/CogVideoX-5b",
            "THUDM/CogVideoX-5b-I2V"
        ]
        
        for model_path in models_to_test:
            info = get_model_info(model_path)
            vram_from_utils = estimate_vram_requirement(model_path)
            
            assert info["vram_gb"] == vram_from_utils, \
                f"VRAM mismatch for {model_path}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_none_model_path(self):
        """None model path should raise ValueError."""
        with pytest.raises((ValueError, AttributeError)):
            get_model_info(None)
    
    def test_very_long_path(self):
        """Very long path should still work if model name is valid."""
        long_path = "/very/long/path/to/models/THUDM/CogVideoX-5b/checkpoint"
        info = get_model_info(long_path)
        assert info["type"] == "t2v"
    
    def test_path_with_version_suffix(self):
        """Path with version suffix should work."""
        info = get_model_info("THUDM/CogVideoX-5b-v1.0")
        # Should still extract cogvideox-5b
        assert info is not None


class TestTypeDetection:
    """Tests for automatic type detection."""
    
    def test_detects_t2v_models(self):
        """Should detect T2V models correctly."""
        t2v_models = [
            "THUDM/CogVideoX-2b",
            "THUDM/CogVideoX-5b",
            "THUDM/CogVideoX1.5-5b"
        ]
        for model in t2v_models:
            info = get_model_info(model)
            assert info["type"] == "t2v", f"{model} should be T2V"
    
    def test_detects_i2v_models(self):
        """Should detect I2V models correctly."""
        i2v_models = [
            "THUDM/CogVideoX-5b-I2V",
            "THUDM/CogVideoX1.5-5b-I2V"
        ]
        for model in i2v_models:
            info = get_model_info(model)
            assert info["type"] == "i2v", f"{model} should be I2V"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
