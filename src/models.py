import torchvision.models as models
from transformers import ViTForImageClassification
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module): # for testing purposes only
    # 1. Dodajemy parametr input_shape (domyślnie dla obrazków 32x32 z 3 kanałami RGB)
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(SmallCNN, self).__init__()
        
        # Warstwy konwolucyjne zostają bez zmian
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def ResNet152():

    model = models.resnet152(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def ConvNeXtLarge():

    model = models.convnext_large(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier[2] = nn.Linear(1536, 10)
    
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

def VisionTransformer():

    model = ViTForImageClassification.from_pretrained(
        "google/vit-huge-patch14-224-in21k",
        num_labels=10
    )
    for param in model.vit.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model

def get_model(model):

    if model == "ResNet":
        return ResNet152()
    elif model == "ConvNeXt":
        return ConvNeXtLarge()
    elif model == "VisionTransformer":
        return VisionTransformer()
    elif model == "SmallCNN":
        return SmallCNN()
    else:
        raise ValueError('You must select one of following model: ["ResNet", "ConvNeXt", "VisionTransformer", "SmallCNN"].')    

def train_mode(model):
    # tryb treningowy
    model.train()

    # zamrażasz backbone
    for param in model.vit.parameters():
        param.requires_grad = False
