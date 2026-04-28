import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(model_name: str, img_size: tuple, split: str = "train") -> tuple:
    """
    Build albumentations transform pipelines for training and validation.

    Satellite images receive an additional random 90-degree rotation during
    training to account for arbitrary north-alignment in aerial imagery.

    Args:
        model_name: HuggingFace model name used to select normalisation stats.
        img_size:   (height, width) tuple for resizing.
        split:      "train" returns (sat_transform, ground_transform);
                    any other value returns (val_transform, val_transform).

    Returns:
        Tuple of (satellite_transform, ground_transform).
    """
    h, w = img_size

    if "clip" in model_name.lower():
        mean, std = _CLIP_MEAN, _CLIP_STD
    else:
        mean, std = _IMAGENET_MEAN, _IMAGENET_STD

    normalize = A.Normalize(mean=mean, std=std)
    resize = A.Resize(h, w, interpolation=cv2.INTER_LINEAR_EXACT)

    if split == "train":
        color_jitter = A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5)
        blur_sharpen = A.OneOf([A.AdvancedBlur(p=1.0), A.Sharpen(p=1.0)], p=0.3)

        sat_transform = A.Compose([
            resize, color_jitter, blur_sharpen,
            A.RandomRotate90(p=1.0),
            normalize, ToTensorV2(),
        ])
        ground_transform = A.Compose([
            resize, color_jitter, blur_sharpen,
            normalize, ToTensorV2(),
        ])
        return sat_transform, ground_transform

    val_transform = A.Compose([resize, normalize, ToTensorV2()])
    return val_transform, val_transform
