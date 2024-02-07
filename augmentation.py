import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A


# If want to define custom augmentation, you can use this class
# https://github.com/albumentations-team/albumentations/issues/938
class SobelFilter(ImageOnlyTransform):
    def __init__(self, prob: float=0.5):
        super().__init__(self)
        self.prob = prob
    
    def apply(self, img, copy=True, **params):
        if np.random.uniform(0, 1) > self.prob:
            return img
        if copy:
            img = img.copy()
            
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        filter = cv2.magnitude(dx, dy)
        filter = cv2.normalize(filter, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return filter
        
        
if __name__ == '__main__':
    img = cv2.imread("data/train/DCM/ID001/image1661130828152_R.png", cv2.IMREAD_GRAYSCALE)
    sobel = SobelFilter(prob=1)
    result = sobel(image=img)
    print(result)