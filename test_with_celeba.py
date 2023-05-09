import torch
from torchvision.datasets import CelebA

dataset = CelebA(root="data", split="train", download=True, target_type="attr")

trainloader = torch.utils.data.DataLoader(dataset)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for idx, (imgs, labels) in enumerate(trainloader):
    imgs = imgs.to(device)
    labels = labels.to(device)
