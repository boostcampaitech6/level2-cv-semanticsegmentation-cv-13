import yaml
import wandb
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
    model.train(
        data=config_path,
        epochs=train_option["epochs"],
        imgsz=train_option["imgsz"],
        device=train_option["device"],
        batch=train_option["batch"],
        workers=train_option["workers"],
    )
    
    
if __name__ == '__main__':
    train('yolo_config.yaml')