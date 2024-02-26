import os
import cv2
import json
import shutil

import yaml
import numpy as np
import pandas as pd
from glob import glob
from skimage import measure
from tqdm import tqdm


# rle를 decode하는 함수를 정의합니다.
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def get_folder_label_name(root_path: str):
    test_img_paths = sorted(glob(os.path.join(root_path, "test", "*", "*", "*.png")))
    test_json_paths = sorted(glob(os.path.join(root_path, "test", "*", "*", "*.json")))
    test_filenames = [x.split('DCM/')[-1] for x in test_img_paths]
    test_labelnames = [x.split('outputs_json/')[-1] for x in test_json_paths]
    
    # test_filename: folder_name/file_name.png
    for test_filename, test_labelname in zip(test_filenames, test_labelnames):
        try:
            assert test_filename.replace(".png", ".json") == test_labelname, f"{test_filename} != {test_labelname}"
        except:
            import sys
            sys.exit(0)
    
    return test_filenames, test_labelnames


def copy_test_data_to_train_path(root_path: str):
    print("Starting copy test images to train path")
    test_img_paths = glob(os.path.join(root_path, "test", "*", "*", "*.png"))
    test_json_paths = glob(os.path.join(root_path, "test", "*", "*", "*.json"))
    
    # make dirs
    test_img_folders = list(set([img_path.split("DCM/")[1].split("/")[0] for img_path in test_img_paths]))
    for test_img_folder in test_img_folders:
        os.makedirs(os.path.join(root_path, "train/DCM", test_img_folder), exist_ok=True)
        os.makedirs(os.path.join(root_path, "train/outputs_json", test_img_folder), exist_ok=True)
    
    # copy
    for test_img_path in test_img_paths:
        shutil.copy(
            test_img_path,
            os.path.join(root_path, "train", test_img_path.split("test/")[1])
        )
        
    for test_json_path in test_json_paths:
        shutil.copy(
            test_json_path,
            os.path.join(root_path, "train", test_json_path.split("test/")[1])
        )    
    print("Finish copy test images to train path")
    print("total_train_images: ", len(glob(os.path.join(root_path, "train", "*", "*", "*.png"))))
    print("total_train_jsons: ", len(glob(os.path.join(root_path, "train", "*", "*", "*.json"))))


def clear_test_data_in_train_path(root_path: str):
    print("Starting remove test images to train path")
    test_img_paths = glob(os.path.join(root_path, "test", "*", "*", "*.png"))
    test_img_folders = list(set([img_path.split("DCM/")[1].split("/")[0] for img_path in test_img_paths]))
    
    for test_img_folder in test_img_folders:
        shutil.rmtree(os.path.join(root_path, "train", "DCM", test_img_folder), ignore_errors=True)
        shutil.rmtree(os.path.join(root_path, "train", "outputs_json", test_img_folder), ignore_errors=True)

    print("Finish remove test images to train path")
    print("total_train_images: ", len(glob(os.path.join(root_path, "train", "*", "*", "*.png"))))
    print("total_train_jsons: ", len(glob(os.path.join(root_path, "train", "*", "*", "*.json"))))
    

def preprocess(
    root_path: str,
    output_csv_path: str, 
):
    # prepare paths
    res_df = pd.read_csv(output_csv_path)
    test_img_names =  res_df['image_name'].unique().tolist()
    test_img_paths = glob(os.path.join(root_path, "test", "*", "*", "*.png"))
    test_json_root = f"{root_path}/test/outputs_json"
    
    if not os.path.exists(test_json_root):
        os.makedirs(test_json_root, exist_ok=True)
    else:
        shutil.rmtree(test_json_root)
        os.makedirs(test_json_root, exist_ok=True)
        
    path_dict = dict()
    for img_path in test_img_paths:
        img_name = img_path.split("/")[-1]
        path_dict[img_name] = img_path
        
    print("Starting convert output.csv to json")
    #  make test json
    for _, test_img_name in tqdm(enumerate(test_img_names)):
        
        # prepare path
        try:
            test_img_path = path_dict[test_img_name]
        except:
            continue
        test_img_folder = test_img_path.split("DCM/")[1].split("/")[0]
        test_json_name = test_img_name.replace(".png", ".json")
        os.makedirs(os.path.join(test_json_root, test_img_folder), exist_ok=True)
        
        # get polygons from output.csv
        img_df = res_df[res_df['image_name'] == test_img_name]

        new_json = {}
        new_annots = []
        filename= test_img_name
        
        for idx, res in enumerate(img_df.values): 
            _, cls_name, rle = res
            try:    
                decoded_rle = rle_decode(rle, (2048, 2048))*255
            except:
                decoded_rle = np.zeros((2048, 2048), dtype=np.uint8)*255
                
            contours = measure.find_contours(decoded_rle, 0.5)
            contours = sorted(contours, key=lambda x: len(x), reverse=True)[0]
            contours = np.flip(contours, axis=1).tolist()
            contours = [[round(x) for x in y] for y in contours]
            
            new_annot = {
                "id": "xxx",
                "type": "poly_seg",
                "attributes": {},
                "points": contours,
                "label": cls_name
            }
            new_annots.append(new_annot)
            
        new_json["annotations"] = new_annots
        new_json["filename"] = filename
    
        with open(os.path.join(test_json_root, test_img_folder, test_json_name), 'w') as f:
            json.dump(new_json, f)
    
    print("Finish convert output.csv to json")


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    root_path = config['DATA_ROOT']
    # clear_test_data_in_train_path(root_path)   
    # preprocess(root_path, config['OUTPUT_CSV_PATH'])
    
    
