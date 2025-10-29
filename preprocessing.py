# Preprocessing
# Use PyTorch (torchvision) to load MNIST and CIFAR-10 datasets.
# Normalize pixel values to [0,1] or use standard normalization.
# Clearly document your preprocessing steps.
# load data https://docs.pytorch.org/vision/stable/datasets.html
# normalize [0,1] = transforms.ToTensor() divide pixel values (0–255) by 255 → range becomes 0 → 1
# std normalize = transforms.ToTensor() → transforms.Normalize(mean, std) Shift & scale values so they have mean ≈ 0 and std ≈ 1
# batch size = 64 because we dont want to feed training data all at once but in mini batches
# shuffle training data to samples each epochs to ensure mini batch contains many classes

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    mnist_training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    mnist_test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    img, label = mnist_training_data[0]
    print(img.shape)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar10_training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    cifar10_test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    img, label = cifar10_training_data[0]
    print(img.shape)

    mnist_train, mnist_val = random_split(mnist_training_data, [50000, 10000])
    mnist_training_data_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_val_data_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)
    mnist_test_data_loader = DataLoader(mnist_test_data, batch_size=batch_size, shuffle=False)

    cifar10_train, cifar10_val = random_split(cifar10_training_data, [45000, 5000])
    cifar10_training_data_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    cifar10_val_data_loader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=False)
    cifar10_test_data_loader = DataLoader(cifar10_test_data, batch_size=batch_size, shuffle=False)
    
    return ( mnist_training_data_loader, mnist_val_data_loader, mnist_test_data_loader, 
        cifar10_training_data_loader, cifar10_val_data_loader, cifar10_test_data_loader )
