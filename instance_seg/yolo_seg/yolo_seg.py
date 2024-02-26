import os
import cv2
import yaml
import wandb
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
import pandas as pd
import albumentations as A


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


# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def train(data_config_path: str):   
    with open(data_config_path, 'r') as file:
        data_config = yaml.safe_load(file)
        
    wandb_option = data_config["wandb_option"]
    wandb.init(
        project=wandb_option["project"],
        entity=wandb_option["entity"],
        name=wandb_option["name"],
    )
    
    train_option = data_config["train_option"]
    # custom_augment = A.Compose([
    #     A.CLAHE(p=0.5),
    #     A.Resize(train_option["imgsz"], train_option["imgsz"]),
    #     A.Rotate(limit=30, p=0.5),
    # ])
    model = YOLO("yolov8x-seg.pt")
    
    # want to customize, See under page
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
    model.train(
        data=data_config_path,
        epochs=train_option["epochs"],
        imgsz=train_option["imgsz"],
        device=train_option["device"],
        batch=train_option["batch"],
        workers=train_option["workers"],
        cos_lr=train_option["cos_lr"],
        optimizer=train_option["optimizer"],
        # mosaic=1.0,
        # fliplr=0.0,
        # erasing=0.0,
        # scale=0.0,
        # translate=0.0,
    )

def inference():
    model = YOLO(f"runs/segment/train/weights/best.pt").cuda()
    infer_images = sorted(glob("../data/test/*/*/*.png"))
    
    rles = []
    filename_and_class = []

    for idx, infer_image in tqdm(enumerate(infer_images)):
        result = model.predict(infer_image, imgsz=2048)[0]
        boxes = result.boxes.data.cpu().numpy()
        scores, classes = boxes[:, 4].tolist(), boxes[:, 5].astype(np.uint8).tolist()
        masks = result.masks.xy
        
        datas = [[a, b, c] for a, b, c, in zip(classes, scores, masks)]
        datas = sorted(datas, key=lambda x: (x[0], -x[1]))
        img_name = infer_image.split("/")[-1]
        
        is_checked = [False] * 29
        csv_idx, data_idx = 0, 0
        while data_idx < len(datas):
            c, s, mask_pts = datas[data_idx]
            # 동일한게 있으면 pass
            if is_checked[c]:
                data_idx += 1
                continue
            
            empty_mask = np.zeros((2048, 2048), dtype=np.uint8)
            if c == csv_idx:
                is_checked[c] = True
                pts = [[int(x[0]), int(x[1])] for x in mask_pts]
                cv2.fillPoly(empty_mask, [np.array(pts)], 1)
                rle = encode_mask_to_rle(empty_mask)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}_{img_name}")
                data_idx += 1
            else:
                rle = encode_mask_to_rle(empty_mask)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[csv_idx]}_{img_name}")
            csv_idx += 1
            
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    if not os.path.exists('./result'):                                                           
        os.makedirs('./result')
    f_name = 'yolo_seg_2048_output.csv'
    df.to_csv(os.path.join('result', f_name), index=False)
        
    
if __name__ == '__main__':
    train('config/yolo_config.yaml')
    inference()