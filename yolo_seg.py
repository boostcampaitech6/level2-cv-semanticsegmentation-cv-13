import cv2
import yaml
import wandb
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO


def train(config_path: str):   
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    wandb_option = config["wandb_option"]
    wandb.init(
        project=wandb_option["project"],
        entity=wandb_option["entity"],
        name=wandb_option["name"],
    )
    
    model = YOLO("yolov8x-seg.pt")
    train_option = config["train_option"]
    print(train_option["close_mosaic"])
    
    # want to customize, See under page
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
    model.train(
        data=config_path,
        epochs=train_option["epochs"],
        imgsz=train_option["imgsz"],
        device=train_option["device"],
        batch=train_option["batch"],
        workers=train_option["workers"],
        close_mosaic=train_option["close_mosaic"],
        cos_lr=train_option["cos_lr"],
        optimizer=train_option["optimizer"],
    )

def inference():
    model = YOLO(f"runs/segment/train/weights/best.pt").cuda()
    infer_images = sorted(glob("data/test/*/*/*.png"))
    
    rles = []
    filename_and_class = []
    for idx, infer_image in tqdm(enumerate(infer_images)):
        result = model.predict(infer_image, imgsz=2048, conf=0.5)[0]
        res = result.plot()
        cv2.imwrite(f"result_{idx}.png", res)   
        
        boxes = result.boxes
        masks = result.masks
        print(boxes, masks)
        break
        
        
        
        
        
    pass

    
if __name__ == '__main__':
    # train('yolo_config.yaml')
    inference()