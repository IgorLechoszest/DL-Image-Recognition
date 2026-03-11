import torchvision.models as models
from transformers import ViTForImageClassification
import torch
import torch.nn as nn

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