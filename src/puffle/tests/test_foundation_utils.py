import torch

from puffle.Utils.constants import DEFAULT_ESTIMATION, EPSILON
from puffle.Utils.tensor_utils import ensure_tensor, safe_divide


def test_constants_values():
    """Verify core constants have expected values."""
    assert DEFAULT_ESTIMATION == 0.5
    assert EPSILON == 1e-10


def test_ensure_tensor_conversion():
    """Test converting list to tensor."""
    data = [1, 2, 3]
    device = "cpu"
    tensor = ensure_tensor(data, device)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == "cpu"
    assert torch.equal(tensor, torch.tensor([1, 2, 3]))


def test_ensure_tensor_idempotency():
    """Test that passing a tensor returns the same tensor if possible."""
    device = "cpu"
    tensor = torch.tensor([1, 2, 3], device=device, dtype=torch.long)
    result = ensure_tensor(tensor, device)
    assert result is tensor


def test_ensure_tensor_device_transfer():
    """Test device transfer in ensure_tensor."""
    # Only test if CUDA is available, otherwise skip or test CPU to CPU with diff dtype
    tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    result = ensure_tensor(tensor, "cpu", dtype=torch.long)
    assert result.dtype == torch.long
    assert result is not tensor


def test_safe_divide():
    """Test safe division with numerical stability."""
    numerator = torch.tensor([1.0, 2.0])
    denominator = torch.tensor([0.0, 0.0])
    result = safe_divide(numerator, denominator)
    # 1.0 / (0.0 + 1e-10) = 1e10
    assert torch.allclose(result, torch.tensor([1e10, 2e10]))


def test_safe_divide_normal():
    """Test safe division with normal values."""
    numerator = torch.tensor([1.0, 4.0])
    denominator = torch.tensor([2.0, 2.0])
    result = safe_divide(numerator, denominator)
    # Should be very close to 0.5 and 2.0
    assert torch.allclose(result, torch.tensor([0.5, 2.0]), atol=1e-7)
