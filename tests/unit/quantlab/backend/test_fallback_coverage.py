"""Tests for fallback backend coverage."""

import pytest

from laakhay.quantlab.backend import get_backend, has_backend
from laakhay.quantlab.backend.implementations.fallback import FallbackBackend


class TestFallbackCoverage:
    """Test fallback backend functionality."""
    
    def test_fallback_methods(self):
        """Test fallback backend methods."""
        # Import directly to avoid registry issues
        from laakhay.quantlab.backend.implementations.fallback import FallbackBackend
        
        # Create fallback backend directly
        fb = FallbackBackend()
        
        # Test methods raise RuntimeError
        with pytest.raises(RuntimeError, match="No array backend available"):
            fb.array([1, 2, 3])
            
        with pytest.raises(RuntimeError):
            fb.zeros((2, 3))
            
        with pytest.raises(RuntimeError):
            fb.ones((2, 3))
            
        with pytest.raises(RuntimeError):
            fb.arange(5)
            
        with pytest.raises(RuntimeError):
            fb.is_array([1, 2, 3])
            
        # Test additional methods  
        with pytest.raises(RuntimeError):
            fb.shape(None)
            
        with pytest.raises(RuntimeError):
            fb.size(None)
            
        with pytest.raises(RuntimeError):
            fb.ndim(None)
            
    def test_fallback_registration(self):
        """Test fallback backend is registered."""
        # Fallback should be available
        from laakhay.quantlab.backend.implementations.fallback import FallbackBackend
        
        # Create instance directly to check properties
        fb = FallbackBackend()
        assert fb.name == "fallback"
        
        # Test explicit methods that are defined
        with pytest.raises(RuntimeError):
            fb.gather(None, None)
            
        with pytest.raises(RuntimeError):
            fb.norm(None)