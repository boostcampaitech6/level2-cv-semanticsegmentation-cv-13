# 손목 X-Ray Segmentation 대회


## 개요

### 배경
뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다. Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로 딥러닝 기술을 이용한 뼈 Segmentation은 질병의 진단 및 치료, 수술 계획, 의료장비 제작, 의료 교육 등에 도움을 줄 수 있습니다. 

### 목적
손가락/손등/팔 29개의 뼈 종류로 구성된 손 X-Ray 데이터로 모델을 학습시켜 각각의 class를 예측하는 Semantic Segmentation을 수행하는 모델을 개발하는 것을 목표로 합니다. 

### 데이터셋과 평가 방법
- **Input**: hand bone x-ray 객체가 담긴 이미지 
- **Annotations**: segmentation annotation정보는 json file로 제공
- **Output**: 모델은 각 클래스(29개)에 대한 멀티 클래스 예측을 수행하고, 예측된 결과를 Run-Length Encoding(RLE) 형식으로 변환하여 csv 파일로 제출

[Wrap up Report](https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-13/blob/develop/Semantic%20Segmentation%20Wrap%20Remove.pdf)


## Project Structure

```
📦level2-cv-semanticsegmentation-cv-13
┣ 📂 eda
┃ ┣ 📜 augmentation_vis.ipynb
┃ ┣ 📜 coco_data_vis.ipynb
┃ ┣ 📜 random_vis.ipynb
┃ ┣ 📜 res_vis.ipynb
┣ 📂 instance_seg
┣ 📂 mmseg
┣ 📜 augmentation.py
┣ 📜 dataloader.py
┣ 📜 inference.py
┣ 📜 loss.py
┣ 📜 model.py
┣ 📜 optimizer.py
┣ 📜 psuedo_label.py
┣ 📜 requirements.txt
┣ 📜 scheduler.py
┣ 📜 train.py
┣ 📜 config.yaml
┗ 📜 README.md
```

## 실행 방법

config.yaml을 원하는 실험의 값으로 변경하여 실험을 진행

> ### data root, saved dir 및 hyperparameter 변경
```
DATA_ROOT: "/data/ephemeral/home/level2-cv-semanticsegmentation-cv-13/data"
SAVED_DIR: "/data/ephemeral/home/level2-cv-semanticsegmentation-cv-13/model"
EXP_NAME: "temp"
CAMPER_ID: "T0000"
BATCH_SIZE: 8
LR: 0.0001
RANDOM_SEED: 21
NUM_EPOCHS: 50
VAL_EVERY: 5
RESIZE: 512
```
> ### Pseudo Labeling 실험 유무
```
PSEUDO_LABEL: False # True : 포함, False : 미포함
# OUTPUT_CSV_PATH : Pseudo 에 쓰일 output.csv 경로
OUTPUT_CSV_PATH: "/data/ephemeral/home/level2-cv-semanticsegmentation-cv-13/result/output.csv"
```
> ### augmentation, loss, optimizer, scheduler 값 변경
```
# augmentation.py에 원하는 aug 추가 후 해당 class 이름으로 변경
augmentation:
  name: "custom"
  params: null

loss:
  name: "bce_dice"
  params: null

# adam, adamw, rmsprop, lion
optimizer:
  name: "adam"
  params: null

# step, cosine, plateau
scheduler:
  name: ""
  params: null
```
> ### model, encoder 변경
```
# torchvision or smp
TYPE: smp

# torchvision: fcn, deeplab
# smp: Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN
MODEL: UnetPlusPlus

# 아래 링크에서 encoder 탐색
# https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file#encoders
ENCODER: efficientnet-b0
```
> ### 학습 진행
```
python train.py
```

## 최종 선택 모델


<p align = "center"> <img height="300px" width="600px" src="https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-13/assets/87365666/3430a29a-866c-4e3f-8d25-b90fa5284cd5"> <p/>

### Hard Voting Ensemble of 3 Models

1. **K-fold Ensemble of Unet++ VGG19**
    - 안정적인 학습과 높은 일반화 성능을 위해 K-fold Ensemble을 진행하였고, 최종 앙상블에도 선택하게 되었습니다.
2. **Unet++ HRNet_w64**
    - 큰 이미지 사이즈(1536 x 1536)로 학습하고 CLAHE 등 뼈의 윤곽을 잘 드러내는 증강 기법을 사용한 모델로, 뼈의 테두리 부분을 특히 잘 잡는 경향이 있어 최종 앙상블에 선택하게 되었습니다.
3. **YOLOv8 Instance Segmentation**
    - 추론 결과 완전히 못 잡는 뼈가 몇 개 있었습니다. 그래서 비록 Semantic Segmentation Task이지만 바운딩 박스를 같이 사용하는 Instance Segmentation을 사용하면 뼈를 놓치지 않고 잡을 수 있다는 가설으로 실험하였고, 실제로 그러한 경향을 보였으며 또한 겹치는 부분에 대한 뼈를 잘 잡는 경향도 있어 최종 앙상블에 선택하게 되었습니다.

### 최종 순위
- Public Score
<p align = "center"> <img src="https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-13/assets/87365666/4db3df1a-489d-4290-a7ab-8a9b1acb22fa"> <p/>

- Private Score
<p align = "center"> <img src="https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-13/assets/87365666/fffa8f31-e636-4b14-90ea-cf6785edb474"> <p/>

---

## Team SMiLE

|    | 김영일_T6030 | 안세희_T6094 | 유한준_T6106 | 윤일호_T6110 | 이재혁_T6132 |
|---|        ---        |        ---        |        ---        |          ---      |        ---        |
|Github|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/patrashu)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/seheeAn)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lukehanjun)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yuniroro)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/NewP1)|
|E-mail|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=qhdrmfdl123@gmail.com)](mailto:qhdrmfdl123@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=imash0525@gmail.com)](mailto:imash0525@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=lukehanjun@gmail.com)](mailto:lukehanjun@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=ilho7159@gmail.com)](mailto:ilho7159@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=jaehyuk712@gmail.com)](mailto:jaehyuk712@gmail.com)|
