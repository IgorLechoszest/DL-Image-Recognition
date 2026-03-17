import torchvision.models as models
from transformers import ViTForImageClassification
import torch.nn as nn
import torch.nn.functional as F
import torch

class SmallCNN(nn.Module): # for testing purposes only
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)


        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def ResNet152():

    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


def ConvNeXtLarge():

    model = models.convnext_large(pretrained=True)
    model.classifier[2] = nn.Linear(1536, 10)

    return model

def VisionTransformer():

    model = ViTForImageClassification.from_pretrained(
        "google/vit-huge-patch14-224",
        num_labels=10
    )

    return model

def get_model(model):
    if model == "ResNet":
        return SmallCNN()#ResNet152()
    elif model == "ConvNeXt":
        return ConvNeXtLarge()
    elif model == "VisionTransformer":
        return VisionTransformer()