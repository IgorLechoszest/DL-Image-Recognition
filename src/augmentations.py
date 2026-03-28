from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import default_collate

def basic_transforms(augmentation_type=None, model_type=None):
    transformations = []

    if augmentation_type == 'flip':
        transformations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            ]
    
    elif augmentation_type == 'shift':
        transformations = [
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2))]
    
    elif augmentation_type == 'rotation':
        transformations = [
        transforms.RandomRotation(15)]
    
    else:
        transformations = []
    
    if model_type in ["ResNet", "DenseNet121", "VGG16_BN"]:
        transformations.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    elif model_type == "SmallCNN":
        transformations.extend([
            # CINIC-10 ma 32x32, więc nie robimy Resize
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4789, 0.4723, 0.4305],
                std=[0.2421, 0.2383, 0.2587]
            )
        ])

    else:
        transformations.extend([transforms.ToTensor()])
    

    return transforms.Compose(transformations)


def cutmix_collate_fn(batch):
    NUM_CLASSES = 10
    cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    return cutmix(*default_collate(batch))