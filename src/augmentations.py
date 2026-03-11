from torchvision import transforms

def basic_transforms():
    
    # TODO: add better augmentation logic

    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])