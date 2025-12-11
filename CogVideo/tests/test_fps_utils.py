"""
Test suite for fps_utils module.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))

from fps_utils import get_correct_fps, validate_fps_for_model, FPS_MAP


class TestGetCorrectFPS:
    """Tests for get_correct_fps function."""
    
    def test_cogvideox_2b(self):
        """CogVideoX-2b should return 8fps."""
        assert get_correct_fps("THUDM/CogVideoX-2b", 49) == 8
    
    def test_cogvideox_5b(self):
        """CogVideoX-5b should return 8fps."""
        assert get_correct_fps("THUDM/CogVideoX-5b", 49) == 8
    
    def test_cogvideox_5b_i2v(self):
        """CogVideoX-5b-I2V should return 8fps."""
        assert get_correct_fps("THUDM/CogVideoX-5b-I2V", 49) == 8
    
    def test_cogvideox1_5_5b(self):
        """CogVideoX1.5-5b should return 16fps."""
        assert get_correct_fps("THUDM/CogVideoX1.5-5b", 81) == 16
    
    def test_cogvideox1_5_5b_i2v(self):
        """CogVideoX1.5-5b-I2V should return 16fps."""
        assert get_correct_fps("THUDM/CogVideoX1.5-5b-I2V", 81) == 16
    
    def test_local_model_path_2b(self):
        """Local path with cogvideox-2b should return 8fps."""
        assert get_correct_fps("./models/cogvideox-2b", 49) == 8
    
    def test_local_model_path_1_5(self):
        """Local path with cogvideox1.5-5b should return 16fps."""
        assert get_correct_fps("./models/cogvideox1.5-5b", 81) == 16
    
    def test_unknown_model_defaults_to_8fps(self):
        """Unknown models should default to 8fps."""
        assert get_correct_fps("unknown/model", 49) == 8
    
    def test_49_frames_uses_8fps(self):
        """49 frames (CogVideoX) should use 8fps."""
        assert get_correct_fps("THUDM/CogVideoX-5b", 49) == 8
    
    def test_81_frames_uses_correct_fps(self):
        """81 frames should detect model type correctly."""
        # CogVideoX with 81 frames (unusual) still gets 8fps
        assert get_correct_fps("THUDM/CogVideoX-5b", 81) == 8
        # CogVideoX1.5 with 81 frames gets 16fps
        assert get_correct_fps("THUDM/CogVideoX1.5-5b", 81) == 16
    
    def test_161_frames_cogvideox1_5(self):
        """161 frames (CogVideoX1.5 long) should use 16fps."""
        assert get_correct_fps("THUDM/CogVideoX1.5-5b", 161) == 16


class TestValidateFPSForModel:
    """Tests for validate_fps_for_model function."""
    
    def test_correct_fps_cogvideox(self):
        """Correct FPS for CogVideoX should validate."""
        is_valid = validate_fps_for_model("THUDM/CogVideoX-5b", 8)
        assert is_valid is True
    
    def test_correct_fps_cogvideox1_5(self):
        """Correct FPS for CogVideoX1.5 should validate."""
        is_valid = validate_fps_for_model("THUDM/CogVideoX1.5-5b", 16)
        assert is_valid is True
    
    def test_incorrect_fps_cogvideox(self):
        """Incorrect FPS for CogVideoX should fail validation."""
        is_valid = validate_fps_for_model("THUDM/CogVideoX-5b", 16)
        assert is_valid is False
    
    def test_incorrect_fps_cogvideox1_5(self):
        """Incorrect FPS for CogVideoX1.5 should fail validation."""
        is_valid = validate_fps_for_model("THUDM/CogVideoX1.5-5b", 8)
        assert is_valid is False


class TestFPSMap:
    """Tests for FPS_MAP constant."""
    
    def test_fps_map_has_cogvideox(self):
        """FPS_MAP should include CogVideoX entries."""
        assert "cogvideox-2b" in FPS_MAP
        assert "cogvideox-5b" in FPS_MAP
    
    def test_fps_map_has_cogvideox1_5(self):
        """FPS_MAP should include CogVideoX1.5 entries."""
        assert "cogvideox1.5-5b" in FPS_MAP
    
    def test_fps_map_values(self):
        """FPS_MAP values should be correct."""
        assert FPS_MAP["cogvideox-2b"] == 8
        assert FPS_MAP["cogvideox-5b"] == 8
        assert FPS_MAP["cogvideox1.5-5b"] == 16
    
    def test_fps_map_has_i2v_variants(self):
        """FPS_MAP should include I2V variants."""
        assert "cogvideox-5b-i2v" in FPS_MAP
        assert "cogvideox1.5-5b-i2v" in FPS_MAP


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_model_path(self):
        """Empty model path should default to 8fps."""
        assert get_correct_fps("", 49) == 8
    
    def test_very_long_model_path(self):
        """Very long model path should still work."""
        long_path = "/very/long/path/to/THUDM/CogVideoX-5b/checkpoint"
        assert get_correct_fps(long_path, 49) == 8
    
    def test_case_variations(self):
        """Different case variations should work."""
        assert get_correct_fps("THUDM/cogvideox-5b", 49) == 8
        assert get_correct_fps("thudm/CogVideoX-5b", 49) == 8
    
    def test_underscore_vs_dash(self):
        """Underscore vs dash should be normalized."""
        # Should normalize cogvideox_5b to cogvideox-5b
        assert get_correct_fps("THUDM/CogVideoX_5b", 49) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
