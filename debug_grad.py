import torch
import torch.nn as nn
from competitors.mmd_fair.model import MMDFairModel
from puffle.Utils.config import PUFFLEConfig
import copy
from competitors.mmd_fair.model import distance_kernel

# Set random seed
torch.manual_seed(42)

model = nn.Linear(5, 1)
x = torch.randn(10, 5)
z = torch.tensor([0,0,0,0,0, 1,1,1,1,1])
y = torch.tensor([1,1,1,1,1, 0,0,0,0,0], dtype=torch.float32)

y_0 = torch.zeros(100)  # Make them very distinct to ensure penalty != 0
y_1 = torch.ones(100)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
mmd = MMDFairModel(model=model, optimizer=optimizer, criterion=nn.BCEWithLogitsLoss(), device='cpu', config=PUFFLEConfig(lambda_regularization=0.0))
mmd.lambda_regularization = 100.0  
mmd.set_tracking_function(y_0, y_1)
mmd.set_client_weights(alpha_0=1.0, alpha_1=1.0)

# manually trace the batch logic
outputs = model(x)
task_loss = nn.BCEWithLogitsLoss()(outputs.squeeze(-1), y)
probs = torch.sigmoid(outputs).squeeze(-1)

mask_0 = z == 0
mask_1 = z == 1
tracking_0 = probs[mask_0]
tracking_1 = probs[mask_1]

c_0 = mmd.tracking_function(tracking_0, demographic_group=0)
c_1 = mmd.tracking_function(tracking_1, demographic_group=1)
fairness_penalty = c_0 - c_1

loss = task_loss + 2 * mmd.lambda_regularization * fairness_penalty

print(f"Task loss: {task_loss.item()}")
print(f"Fairness penalty: {fairness_penalty.item()}")
print(f"Total loss: {loss.item()}")

loss.backward()
print("Gradients:", model.weight.grad)
