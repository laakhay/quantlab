"""Device management."""

from __future__ import annotations


class Device:
    """Compute device (CPU/GPU)."""
 
    def __init__(self, device_type: str, device_id: int = 0):
        self.type = device_type.lower()
        if self.type == 'gpu':
            self.type = 'cuda'
        self.id = device_id

    def __str__(self):
        if self.type == 'cpu':
            return 'cpu'
        return f'{self.type}:{self.id}'

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        return self.type == other.type and self.id == other.id
    
    def __hash__(self):
        return hash((self.type, self.id))

    @classmethod
    def cpu(cls):
        """CPU device."""
        return cls('cpu')

    @classmethod
    def gpu(cls, device_id: int = 0):
        """GPU device."""
        return cls('cuda', device_id)


def get_device(array) -> Device:
    """Infer device from array."""
    if hasattr(array, 'device') and callable(array.device):
        device = array.device()
        if hasattr(device, 'platform'):
            if device.platform == 'gpu':
                return Device.gpu(device.id)
            return Device.cpu()

    if hasattr(array, 'device') and not callable(array.device):
        device = array.device
        if hasattr(device, 'type') and device.type == 'cuda':
            return Device.gpu(device.index or 0)
        return Device.cpu()

    return Device.cpu()