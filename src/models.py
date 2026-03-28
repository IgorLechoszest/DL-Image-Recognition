import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch

class SmallCNN(nn.Module): 
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(SmallCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc1 = nn.Linear(64 * 4 * 4, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x) 
        x = self.fc2(x)

        return x

def ResNet152(dropout_rate=0.2):
    model = models.resnet152(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.fc.in_features, 10)
    )
    
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def DenseNet121(dropout_rate=0.2):
    model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.classifier.in_features, 10)
    )
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def VGG16_BN(dropout_rate=0.5):
    model = models.vgg16_bn(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier[6] = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.classifier[6].in_features, 10)
    )
    
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    return model

def get_model(model_name, dropout_rate=0.2):
    if model_name == "ResNet":
        return ResNet152(dropout_rate)
    elif model_name == "SmallCNN":
        return SmallCNN(dropout_rate=dropout_rate)
    elif model_name == "DenseNet121":
        return DenseNet121(dropout_rate)
    elif model_name == "VGG16_BN":
        return VGG16_BN(dropout_rate)
    else:
        raise ValueError('You must select one of following model: ["ResNet", "SmallCNN", "DenseNet121", "VGG16_BN"].')