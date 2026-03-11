from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(transform, batch_size=128):

    train_dataset = datasets.ImageFolder(
        "data/CINIC-10/train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        "data/CINIC-10/valid",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        "data/CINIC-10/test",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader