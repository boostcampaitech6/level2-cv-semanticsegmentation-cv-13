import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import segmentation_models_pytorch as smp
# import timm

class FcnResnet50(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        # output class 개수를 dataset에 맞도록 수정합니다.
        self.model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    
    def forward(self, x):
        output = self.model(x)['out']
        return output

class DeepLabv3Resnet50(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, len(CLASSES), kernel_size=1)

    def forward(self, x):
        output = self.model(x)['out']
        return output

class SMPModel(nn.Module):
    def __init__(self, model_name, encoder, CLASSES):
        super().__init__()

        self.model = smp.create_model(
            model_name,
            encoder_name = encoder, 
            encoder_weights="imagenet",     
            in_channels=3,                  
            classes=len(CLASSES),   
        )

    def forward(self, x):
        output = self.model(x)
        return output


# 사용 가능한 모델 함수의 진입점
_model_entrypoints = {
    "torchvision" : {
        "fcn": FcnResnet50,
        "deeplab": DeepLabv3Resnet50
    },
    "smp" : SMPModel
}

def create_model(model_type, model_name, encoder, CLASSES):
    if model_type == "torchvision":
        _model = _model_entrypoints[model_type][model_name]
        model = _model(CLASSES)

    elif model_type == "smp":
        _model = _model_entrypoints[model_type]
        model = _model(model_name, encoder, CLASSES)
    
    else:
        raise RuntimeError("정의되지 않은 타입")
    
    return model