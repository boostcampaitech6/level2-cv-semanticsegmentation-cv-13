import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import segmentation_models_pytorch as smp


class FcnResnet50(nn.Module):
    def __init__(self, CLASSES):
        super().__init__()
        model = models.segmentation.fcn_resnet50(pretrained=True)
        # output class 개수를 dataset에 맞도록 수정합니다.
        model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    
    def forward(self, x):
        output = self.model(x)['out']
        return output


class Unet():
    def __init__(self, CLASSES):
        model = smp.Unet(
            encoder_name="efficientnet-b0", 
            encoder_weights="imagenet",     
            in_channels=3,                  
            classes=len(CLASSES),                     
        )

    def forward(self, x):
        output = self.model(x)
        return output


# 사용 가능한 모델 함수의 진입점
_model_entrypoints = {
    "fcn": FcnResnet50,
    "unet": Unet
}

def create_model(model_type, CLASSES):
    if model_type in _model_entrypoints:
        _model = _model_entrypoints(model_type)
        model = _model(CLASSES)

    else:    
        raise RuntimeError("해당 모델이 정의되지 않았습니다.")
    
    return model