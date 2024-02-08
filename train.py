# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A
import yaml
import wandb

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import segmentation_models_pytorch as smp
from model import create_model

# visualization
import matplotlib.pyplot as plt
from dataloader import XRayDataset
from psuedo_label import *
from augmentation import SobelFilter


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


def save_model(model, file_name='model.pt'):
    output_path = os.path.join(RESULT_DIR, file_name)
    torch.save(model, output_path)


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    set_seed()
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
            print(outputs.shape, masks.shape)
                
            # B C H W            
            if (step+1)%10 == 0:
                table_data = []
                masks, preds = masks[0].numpy(), outputs[0].numpy()
                for cls_idx in range(n_class):
                    empty_mask = np.zeros((2048, 2048))
                    mask = masks[cls_idx].astype(np.uint8) * 64
                    pred = preds[cls_idx].astype(np.uint8) * 128
                    
                    empty_mask += mask
                    empty_mask += pred
                    table_data.append([IND2CLASS[cls_idx], wandb.Image(empty_mask)])
                
                wandb.log({"val/vis_img": wandb.Table(columns=["cls_name", "img"], data=table_data)}, step=epoch)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    wandb.log({"val/loss": total_loss / cnt, "val/dice": avg_dice}, step=epoch)
    for c, d in zip(CLASSES, dices_per_class):
        wandb.log({f"val-class/dice_{c}": d.item()}, step=epoch)
    return avg_dice


def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    set_seed()
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            outputs = model(images)
            
            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 10 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
        
        wandb.log({"train/loss": loss.item()}, step=epoch)

        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    DATA_ROOT = config['DATA_ROOT']
    IMAGE_ROOT = f"{DATA_ROOT}/train/DCM"
    LABEL_ROOT = f"{DATA_ROOT}/train/outputs_json"
    SAVED_DIR = config['SAVED_DIR']
    EXP_NAME = config['EXP_NAME']
    RESULT_DIR = os.path.join(SAVED_DIR, EXP_NAME)

    if not os.path.exists(RESULT_DIR):                                                           
        os.makedirs(RESULT_DIR)

    BATCH_SIZE = config['BATCH_SIZE']
    LR = config['LR']
    RANDOM_SEED = config['RANDOM_SEED']
    NUM_EPOCHS = config['NUM_EPOCHS']
    VAL_EVERY = config['VAL_EVERY']
    PSUEDOLABEL_FLAG = config['PSEUDO_LABEL']

    # model 정의
    TYPE = config['TYPE']
    MODEL = config['MODEL']
    ENCODER = config['ENCODER']
    RESIZE = config['RESIZE']
    
    clear_test_data_in_train_path(DATA_ROOT)
    if PSUEDOLABEL_FLAG:
        preprocess(DATA_ROOT, config['OUTPUT_CSV_PATH'])
    
    # resize
    tf = A.Resize(RESIZE, RESIZE)
    # tf = A.Compose([
    #     SobelFilter(prob=0.5),
    #     A.Resize(RESIZE, RESIZE),
    # ])

    train_dataset = XRayDataset(
        IMAGE_ROOT, 
        LABEL_ROOT, 
        is_train=True, 
        transforms=tf,
        psuedo_flag=PSUEDOLABEL_FLAG,
    )
    valid_dataset = XRayDataset(
        IMAGE_ROOT, 
        LABEL_ROOT, 
        is_train=False, 
        transforms=tf,
    )
    
    if PSUEDOLABEL_FLAG:
        copy_test_data_to_train_path("data")

    print(len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # model을 정의
    model = create_model(TYPE, MODEL, ENCODER, CLASSES)
    
    # Loss function을 정의합니다.
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer를 정의합니다.
    optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

    # 시드를 설정합니다.
    set_seed()

    CAMPER_ID = config['CAMPER_ID']
    wandb.init(project='Boost Camp Lv2-3',entity='frostings', name=f"{CAMPER_ID}-{EXP_NAME}", config=config)
    
    train(model, train_loader, valid_loader, criterion, optimizer)
