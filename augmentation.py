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


# for base experiment setting and validation set
class BaseAugmentation:
    def __init__(self, resize):
        self.transform = A.Resize(resize, resize)

    def __call__(self, **kwargs):
        return self.transform(**kwargs)


# for further augmentation experiments
class CustomAugmentation:
    def __init__(self, resize):
        self.transform = A.Resize(resize, resize)

    def __call__(self, image, mask=None):
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
        else:
            augmented = self.transform(image=image)
        return augmented


# Augmentation Registry
_augmentation_entrypoints = {
    "base": BaseAugmentation,
    "custom": CustomAugmentation
}


def create_augmentation(aug_name, **kwargs):
    if aug_name in _augmentation_entrypoints:
        aug_class = _augmentation_entrypoints[aug_name]
        return aug_class(**kwargs)
    else:
        raise RuntimeError(f"Unknown augmentation ({aug_name})")
