import os
import json

import numpy as np
import cv2
from sklearn.model_selection import GroupKFold

from mmseg.registry import DATASETS, TRANSFORMS, MODELS, METRICS
from mmseg.datasets import BaseSegDataset

from mmcv.transforms import BaseTransform

# 데이터 경로를 입력하세요

IMAGE_ROOT = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-13/data/train/DCM'
LABEL_ROOT = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-13/data/train/outputs_json'

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

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

_filenames = np.array(pngs)
_labelnames = np.array(jsons)
    
# split train-valid
# 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
# 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
# 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
groups = [os.path.dirname(fname) for fname in _filenames]

# dummy label
ys = [0 for fname in _filenames]

# 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
# 5으로 설정하여 KFold를 수행합니다.
gkf = GroupKFold(n_splits=5)

@DATASETS.register_module()
class XRayDataset2(BaseSegDataset):
    def __init__(self, is_train, **kwargs):
        self.is_train = is_train
        
        super().__init__(**kwargs)
    
    # def load_data_list(self):
    #     if self.is_train:
    #         filenames = pngs[160:]
    #         labelnames = jsons[160:]
        
    #     else:
    #         filenames = pngs[:160]
    #         labelnames =jsons[:160]
        
    #     data_list = []
    #     for i, (img_path, ann_path) in enumerate(zip(filenames, labelnames)):
    #         data_info = dict(
    #             img_path=os.path.join(IMAGE_ROOT, img_path),
    #             seg_map_path=os.path.join(LABEL_ROOT, ann_path),
    #         )
    #         data_list.append(data_info)
        
    #     return data_list
        
    def load_data_list(self):
        filenames = []
        labelnames = []
        valid_set_num = 2
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                if i == valid_set_num:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                if i == valid_set_num:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])
                    break
        
        data_list = []
        for i, (img_path, ann_path) in enumerate(zip(filenames, labelnames)):
            data_info = dict(
                img_path=os.path.join(IMAGE_ROOT, img_path),
                seg_map_path=os.path.join(LABEL_ROOT, ann_path),
            )
            data_list.append(data_info)
        
        return data_list
        

@TRANSFORMS.register_module()
class LoadXRayAnnotations(BaseTransform):
    def transform(self, result):
        label_path = result["seg_map_path"]
        
        image_size = (2048, 2048)
        
        # process a label of shape (H, W, NC)
        label_shape = image_size + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image_size, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        result["gt_seg_map"] = label
        
        return result
    
@TRANSFORMS.register_module()
class TransposeAnnotations(BaseTransform):
    def transform(self, result):
        result["gt_seg_map"] = np.transpose(result["gt_seg_map"], (2, 0, 1))
        
        return result