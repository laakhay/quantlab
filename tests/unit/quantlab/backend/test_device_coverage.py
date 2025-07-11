"""Tests for device module coverage."""

import pytest

from laakhay.quantlab.backend.device import Device


class TestDeviceCoverage:
    """Test device functionality comprehensively."""

    def test_device_creation(self):
        """Test device creation."""
        # CPU device
        cpu = Device("cpu")
        assert cpu.type == "cpu"
        assert cpu.id == 0
        assert str(cpu) == "cpu"
        
        # GPU device (converts to cuda)
        gpu = Device("gpu", 0)
        assert gpu.type == "cuda"
        assert gpu.id == 0
        assert str(gpu) == "cuda:0"
        
        # CUDA device
        cuda = Device("cuda", 1)
        assert cuda.type == "cuda"
        assert cuda.id == 1
        assert str(cuda) == "cuda:1"
        
        # Case insensitive
        cpu_upper = Device("CPU")
        assert cpu_upper.type == "cpu"
        
    def test_device_equality(self):
        """Test device equality."""
        cpu1 = Device("cpu")
        cpu2 = Device("cpu", 0)
        assert cpu1 == cpu2
        
        gpu1 = Device("gpu", 0)
        gpu2 = Device("cuda", 0)
        assert gpu1 == gpu2
        
        gpu_diff = Device("gpu", 1)
        assert gpu1 != gpu_diff
        
        assert cpu1 != gpu1
        
    def test_device_repr(self):
        """Test device representation."""
        cpu = Device("cpu")
        assert repr(cpu) == "Device(cpu)"
        
        gpu = Device("gpu", 1)
        assert repr(gpu) == "Device(cuda:1)"
        
    def test_device_hash(self):
        """Test device hashing."""
        cpu1 = Device("cpu")
        cpu2 = Device("cpu")
        assert hash(cpu1) == hash(cpu2)
        
        # Can be used in sets/dicts
        device_set = {cpu1, cpu2}
        assert len(device_set) == 1
        
    def test_device_factory_methods(self):
        """Test device factory methods."""
        # CPU device
        cpu = Device.cpu()
        assert cpu.type == "cpu"
        assert cpu.id == 0
        
        # GPU device
        gpu = Device.gpu(2)
        assert gpu.type == "cuda"
        assert gpu.id == 2
        
        # Default GPU
        gpu_default = Device.gpu()
        assert gpu_default.type == "cuda"
        assert gpu_default.id == 0
        
    def test_device_available(self):
        """Test device availability checks."""
        # CPU is always available
        cpu = Device.cpu()
        # Check if is_available method exists
        if hasattr(cpu, 'is_available'):
            assert cpu.is_available()
        
        # GPU availability depends on system
        gpu = Device.gpu()
        # Just check if method exists
        if hasattr(gpu, 'is_available'):
            assert isinstance(gpu.is_available(), bool)