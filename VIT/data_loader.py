import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import os
from config import get_default_config

# Get batch size from config
config = get_default_config()
batch_size = config.batch_size

# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(15), 
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Just normalization for testing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Download and load datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_data_loaders(distributed_training=False, local_rank=-1):
    """Create and return data loaders based on training mode"""
    if distributed_training:
        train_sampler = DistributedSampler(trainset, shuffle=True)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False,
            sampler=train_sampler, num_workers=4, pin_memory=True
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )

    # Test loader is always the same
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return trainloader, testloader

