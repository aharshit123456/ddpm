from torchvision import transforms 
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_transformed_dataset(IMG_SIZE=64):
    # Define the transformation pipeline
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    # Load CIFAR10 dataset without splitting
    cifar10_dataset = torchvision.datasets.CIFAR10(root=".", download=True, transform=data_transform)

    # Split indices into train and test using sklearn's train_test_split
    dataset_size = len(cifar10_dataset)
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    # Create train and test subsets
    train_subset = Subset(cifar10_dataset, train_indices)
    test_subset = Subset(cifar10_dataset, test_indices)

    # Combine train and test subsets into a single ConcatDataset
    combined_dataset = torch.utils.data.ConcatDataset([train_subset, test_subset])

    return combined_dataset

def load_dataloader(combined_dataset, batch_size=64):
    # Create dataloaders
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader