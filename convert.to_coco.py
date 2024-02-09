import os
import json

import yaml
from glob import glob
from PIL import Image
from tqdm import tqdm


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i+1 for i, v in enumerate(CLASSES)}


def convert_data_to_coco_format(
    root_path: str,
    method: str,
):
    print("Starting convert data to coco format")
    img_paths = sorted(glob(os.path.join(root_path, method, "*", "*", "*.png")))
    json_paths = sorted(glob(os.path.join(root_path, method, "*", "*", "*.json")))
    
    coco_json = {}
    images = []
    annotations = []
    categories = []
    
    # 1 to 29
    for k, v in CLASS2IND.items():
        categories.append({
            "id": v,
            "name": k
        })
    
    annot_idx = 0
    for img_idx, (img_path, json_path) in tqdm(enumerate(zip(img_paths, json_paths))):
        # image와 json name이 잘 mapping 되는지 확인
        # img_path = img_path.replace("\\", "/").split("/")[-1].replace(".png", "")
        # json_path = json_path.replace("\\", "/").split("/")[-1].replace(".json", "")
        # assert img_path == json_path, f"{img_path} != {json_path}"
        
        pil_img = Image.open(img_path)
        width, height = pil_img.size
        file_name = img_path.replace("\\", "/").split(f"{method}/")[-1]
        
        image = {
            "id": img_idx,
            "file_name": file_name,
            "height": width,
            "width": height,
        }
        images.append(image)
        
        with open(json_path, 'r') as f:
            annot_data = json.load(f)
        
        for annot in annot_data["annotations"]:
            annotation = {}
            annotation["id"] = annot_idx
            annotation["image_id"] = img_idx
            annotation["category_id"] = CLASS2IND[annot["label"]]
            
            # get polygon points
            points = annot["points"]
            min_x, min_y = min(points, key=lambda x: x[0])[0], min(points, key=lambda x: x[1])[1]
            max_x, max_y = max(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[1])[1]
            w, h = max_x - min_x, max_y - min_y
            
            bbox = [min_x, min_y, w, h]
            area = w * h
            points = [p for point in points for p in point]
            iscrowd = 0
            
            annotation["bbox"] = bbox
            annotation["area"] = area
            annotation["segmentation"] = [points]
            annotation["iscrowd"] = iscrowd
            annotations.append(annotation)
            annot_idx += 1
            
    print("Finish convert data to coco format")
    
    # save json file
    coco_json["images"] = images
    coco_json["annotations"] = annotations
    coco_json["categories"] = categories
    with open(f"{root_path}/{method}.json", "w") as f:
        json.dump(coco_json, f, indent=4)
    


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    root_path = config['DATA_ROOT']
    convert_data_to_coco_format(root_path, "train")