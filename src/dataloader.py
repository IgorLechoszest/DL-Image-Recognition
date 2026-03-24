from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from collections import defaultdict
import random
import numpy as np

def get_train_dataset(transform):

    train_dataset = datasets.ImageFolder(
        "../data/train",
        transform = transform
    )
    return train_dataset

def get_val_train_dataset(model_type):
    if model_type == "ResNet" or model_type == "ConvNeXt":
        transform = transforms.Compose(
            [transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
            )
        
    elif model_type == "SmallCNN":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4789, 0.4723, 0.4305],
                std=[0.2421, 0.2383, 0.2587]
            )
        ])

    elif model_type == "VisionTransformer":
        transform = transforms.Compose(
            [transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])]
            )
    
    val_dataset = datasets.ImageFolder(
            "../data/valid",
            transform = transform
        )

    test_dataset = datasets.ImageFolder(
            "../data/test",
            transform = transform
        )
    return val_dataset, test_dataset

def get_train_dataloaders(train_dataset, collate_fn=None, batch_size = 128):

    if collate_fn is None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True) 
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True ,collate_fn = collate_fn)
    return train_loader

def get_val_test_dataloaders(val_dataset, test_dataset, batch_size=128):

    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return val_loader, test_loader



def get_subset(dataset, samples_per_class=5, seed=42):


    random.seed(seed)

    class_indices = defaultdict(list)

    for idx, label in enumerate(dataset.targets):
        class_indices[label].append(idx)

    selected_indices = []

    for label, indices in class_indices.items():
        if len(indices) < samples_per_class:
            raise ValueError(f"Not enough samples for label: {label}")

        selected = random.sample(indices, samples_per_class)
        selected_indices.extend(selected)

    return Subset(dataset, selected_indices)
