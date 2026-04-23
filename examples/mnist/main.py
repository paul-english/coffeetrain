#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from coffeetrain import TrainerV2

trainer = TrainerV2()


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 128), nn.ReLU(),   # 64 * 5 * 5 = 1600
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


@trainer.system('DATA_BEFORE')
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    return {
        'train_dataloader': DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True),
        'eval_dataloader': DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4),
        'loss_fn': nn.CrossEntropyLoss(),
    }


@trainer.system('MODEL_BEFORE')
def create_model():
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MnistCNN().to(compute_device)
    # device=None prevents cuda_accelerate.move_to_gpu from calling .to() on MNIST tuples
    return {'model': model, 'device': None, 'compute_device': compute_device}


@trainer.system('OPTIMIZER_BEFORE')
def create_optimizer(model, lr=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return {'optimizer': optimizer, 'lr': lr}


@trainer.system('FORWARD_BEFORE')
def prepare_forward_batch(batch, compute_device):
    inputs, labels = batch
    return {'inputs': inputs.to(compute_device), 'labels': labels.to(compute_device)}


@trainer.system('FORWARD')
def forward_pass(model, inputs):
    return {'outputs': model(inputs)}


@trainer.system('BACKWARD_AFTER')
def optimizer_step(optimizer):
    optimizer.step()
    optimizer.zero_grad()


@trainer.system('EVAL_FORWARD_BEFORE')
def prepare_eval_batch(batch, compute_device):
    inputs, labels = batch
    return {'inputs': inputs.to(compute_device), 'labels': labels.to(compute_device)}


@trainer.system('EVAL_FORWARD')
def eval_forward_pass(model, inputs):
    return {'outputs': model(inputs)}


if __name__ == '__main__':
    trainer()
