import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        
        loss = (1 - ((2. * intersection + self.smooth) / 
                     (pred.sum(dim=2).sum(dim=2) + 
                      target.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()
    

class IOULoss(nn.Module):
    def __init__(self, smooth=1):
        super(IOULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        IoU = (intersection + self.smooth) / (union + self.smooth)
        return 1 - IoU


class BCEPlusDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEPlusDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        dice_loss = self.dice_loss(torch.sigmoid(inputs), targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BCEPlusFocalLoss(nn.Module):
    def __init__(self, bce_weight=0.5, focal_weight=0.5, alpha=0.25, gamma=2):
        super(BCEPlusFocalLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss


class BCEPlusIOULoss(nn.Module):
    def __init__(self, bce_weight=0.5, iou_weight=0.5):
        super(BCEPlusIOULoss, self).__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IOULoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        iou_loss = self.iou_loss(torch.sigmoid(inputs), targets)
        return self.bce_weight * bce_loss + self.iou_weight * iou_loss


# 사용 가능한 손실 함수의 진입점
_criterion_entrypoints = {
    "cross_entropy": nn.BCEWithLogitsLoss,
    "focal": FocalLoss,
    "dice": DiceLoss,
    "iou": IOULoss,
    "bce_dice": BCEPlusDiceLoss,
    "bce_focal": BCEPlusFocalLoss,
    "bce_iou": BCEPlusIOULoss,
}


def create_criterion(criterion_name, **kwargs):
    """
    지정된 인수를 사용하여 손실 함수 객체를 생성한다.

    Args:
        criterion_name (str): 생성할 손실 함수 이름
        **kargs: 손실 함수 생성자에 전달된 키워드 인자

    Returns:
        nn.Module: 생성된 손실 함수 객체
    """
    if criterion_name in _criterion_entrypoints:
        create_fn = _criterion_entrypoints[criterion_name]
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError(f"Unknown loss ({criterion_name})")
    return criterion
