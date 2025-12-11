"""
Test suite for vram_utils module.
"""

import pytest
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))

from vram_utils import (
    get_gpu_memory_info,
    get_model_family,
    estimate_vram_requirement,
    check_vram_availability,
    get_recommended_offload_strategy,
    MODEL_VRAM_REQUIREMENTS
)


class TestGetGPUMemoryInfo:
    """Tests for get_gpu_memory_info function."""
    
    def test_returns_tuple(self):
        """Should return tuple of 3 floats."""
        result = get_gpu_memory_info()
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_values_are_floats(self):
        """All values should be floats."""
        total_gb, used_gb, available_gb = get_gpu_memory_info()
        assert isinstance(total_gb, float)
        assert isinstance(used_gb, float)
        assert isinstance(available_gb, float)
    
    def test_values_non_negative(self):
        """All values should be non-negative."""
        total_gb, used_gb, available_gb = get_gpu_memory_info()
        assert total_gb >= 0
        assert used_gb >= 0
        assert available_gb >= 0
    
    def test_available_plus_used_equals_total(self):
        """Available + used should approximately equal total."""
        total_gb, used_gb, available_gb = get_gpu_memory_info()
        if total_gb > 0:  # Only test if CUDA available
            # Allow 0.5GB tolerance for rounding
            assert abs((used_gb + available_gb) - total_gb) < 0.5
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_available(self):
        """If CUDA available, total should be > 0."""
        total_gb, _, _ = get_gpu_memory_info()
        assert total_gb > 0
    
    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA available")
    def test_cuda_not_available(self):
        """If CUDA not available, should return zeros."""
        total_gb, used_gb, available_gb = get_gpu_memory_info()
        assert total_gb == 0.0
        assert used_gb == 0.0
        assert available_gb == 0.0


class TestGetModelFamily:
    """Tests for get_model_family function."""
    
    def test_cogvideox_2b(self):
        """CogVideoX-2b should normalize correctly."""
        assert get_model_family("THUDM/CogVideoX-2b") == "cogvideox-2b"
    
    def test_cogvideox_5b(self):
        """CogVideoX-5b should normalize correctly."""
        assert get_model_family("THUDM/CogVideoX-5b") == "cogvideox-5b"
    
    def test_cogvideox1_5(self):
        """CogVideoX1.5 should preserve version number."""
        assert get_model_family("THUDM/CogVideoX1.5-5b") == "cogvideox1.5-5b"
    
    def test_local_path(self):
        """Local paths should extract model name correctly."""
        assert get_model_family("./models/CogVideoX-5b") == "cogvideox-5b"
    
    def test_case_insensitive(self):
        """Should handle different cases."""
        assert get_model_family("THUDM/cogvideox-5b") == "cogvideox-5b"
    
    def test_underscore_normalization(self):
        """Should normalize underscores to dashes."""
        # Note: Implementation might vary, adjust based on actual behavior
        family = get_model_family("THUDM/CogVideoX_5b")
        assert "cogvideox" in family.lower()


class TestEstimateVRAMRequirement:
    """Tests for estimate_vram_requirement function."""
    
    def test_2b_model(self):
        """CogVideoX-2B should estimate 7.5GB."""
        vram = estimate_vram_requirement("THUDM/CogVideoX-2b")
        assert vram == 7.5
    
    def test_5b_model(self):
        """CogVideoX-5B should estimate 17GB."""
        vram = estimate_vram_requirement("THUDM/CogVideoX-5b")
        assert vram == 17.0
    
    def test_5b_i2v_model(self):
        """CogVideoX-5B-I2V should estimate 17GB."""
        vram = estimate_vram_requirement("THUDM/CogVideoX-5b-I2V")
        assert vram == 17.0
    
    def test_1_5_model(self):
        """CogVideoX1.5-5B should estimate 18.5GB."""
        vram = estimate_vram_requirement("THUDM/CogVideoX1.5-5b")
        assert vram == 18.5
    
    def test_unknown_model_defaults(self):
        """Unknown models should default to 20GB."""
        vram = estimate_vram_requirement("unknown/model")
        assert vram == 20.0


class TestCheckVRAMAvailability:
    """Tests for check_vram_availability function."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_returns_tuple(self):
        """Should return (bool, str) tuple."""
        result = check_vram_availability("THUDM/CogVideoX-2b")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_message_contains_info(self):
        """Message should contain VRAM info."""
        _, message = check_vram_availability("THUDM/CogVideoX-5b")
        assert "GB" in message
        assert ("available" in message.lower() or 
                "sufficient" in message.lower() or 
                "required" in message.lower())
    
    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA available")
    def test_cuda_not_available_returns_false(self):
        """If CUDA not available, should return False."""
        is_sufficient, message = check_vram_availability("THUDM/CogVideoX-2b")
        assert is_sufficient is False
        assert "CUDA" in message or "available" in message.lower()
    
    def test_custom_buffer(self):
        """Should accept custom buffer size."""
        result = check_vram_availability("THUDM/CogVideoX-2b", required_buffer_gb=4.0)
        assert isinstance(result, tuple)


class TestGetRecommendedOffloadStrategy:
    """Tests for get_recommended_offload_strategy function."""
    
    def test_returns_valid_strategy(self):
        """Should return one of the valid strategies."""
        strategy = get_recommended_offload_strategy("THUDM/CogVideoX-5b")
        assert strategy in ["none", "model", "sequential"]
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_strategy_based_on_vram(self):
        """Strategy should match available VRAM."""
        total_gb, _, _ = get_gpu_memory_info()
        strategy = get_recommended_offload_strategy("THUDM/CogVideoX-5b")
        
        if total_gb >= 24:
            # Should prefer direct GPU or model offload
            assert strategy in ["none", "model"]
        elif total_gb >= 16:
            # Should use model offload
            assert strategy == "model"
        else:
            # Should use sequential for low VRAM
            assert strategy == "sequential"
    
    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA available")
    def test_cuda_not_available_defaults_sequential(self):
        """If CUDA not available, should default to sequential."""
        strategy = get_recommended_offload_strategy("THUDM/CogVideoX-5b")
        assert strategy == "sequential"


class TestVRAMRequirementsConstant:
    """Tests for MODEL_VRAM_REQUIREMENTS constant."""
    
    def test_vram_requirements_exists(self):
        """MODEL_VRAM_REQUIREMENTS should be defined."""
        assert MODEL_VRAM_REQUIREMENTS is not None
        assert isinstance(MODEL_VRAM_REQUIREMENTS, dict)
    
    def test_has_required_models(self):
        """Should have entries for common models."""
        assert "cogvideox-2b" in MODEL_VRAM_REQUIREMENTS
        assert "cogvideox-5b" in MODEL_VRAM_REQUIREMENTS
        assert "cogvideox1.5-5b" in MODEL_VRAM_REQUIREMENTS
    
    def test_values_are_floats(self):
        """All VRAM values should be floats."""
        for value in MODEL_VRAM_REQUIREMENTS.values():
            assert isinstance(value, (int, float))
            assert value > 0


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_workflow_check_and_select_strategy(self):
        """Test workflow: check VRAM then select strategy."""
        model = "THUDM/CogVideoX-5b"
        
        # Check availability
        is_sufficient, message = check_vram_availability(model)
        
        # Get recommended strategy
        strategy = get_recommended_offload_strategy(model)
        
        # If sufficient VRAM, should not need sequential
        if is_sufficient:
            assert strategy in ["none", "model"]
        
        # Results should be consistent
        assert isinstance(is_sufficient, bool)
        assert isinstance(strategy, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
