import torch
from competitors.mmd_fair.model import distance_kernel

# We simulate predictions entering the kernel
# Case 1: logits (large numbers)
# Both > 1
a = torch.tensor([[5.0]], requires_grad=True)
b = torch.tensor([[2.0]])
k = distance_kernel(a, b)
k.backward()
print("Gradient when a=5, b=2:", a.grad)

# Both < 0
a2 = torch.tensor([[-5.0]], requires_grad=True)
b2 = torch.tensor([[-2.0]])
k2 = distance_kernel(a2, b2)
k2.backward()
print("Gradient when a=-5, b=-2:", a2.grad)

# Case 2: probabilities (0 < a, b < 1)
a3 = torch.tensor([[0.8]], requires_grad=True)
b3 = torch.tensor([[0.2]])
k3 = distance_kernel(a3, b3)
k3.backward()
print("Gradient when a=0.8, b=0.2:", a3.grad)
