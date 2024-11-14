import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import torch
import io

from utils import set_random_seed, to_tensor, to_pil

class ImageDistortion(torch.nn.Module):
    def __init__(self, distortion_type, same_operation=False, relative_strength=True, return_image=True):
        super(ImageDistortion, self).__init__()
        self.distortion_type = distortion_type
        self.same_operation = same_operation
        self.relative_strength = relative_strength
        self.return_image = return_image
        # Parameters for distortion strength
        self.distortion_strength_params = {
            "rotation": (0, 45),
            "resizedcrop": (1, 0.5),
            "erasing": (0, 0.25),
            "brightness": (1, 2),
            "contrast": (1, 2),
            "blurring": (0, 20),
            "noise": (0, 0.1),
            "compression": (90, 10),
        }

    def relative_strength_to_absolute(self, strength):
        assert 0 <= strength <= 1
        min_val, max_val = self.distortion_strength_params[self.distortion_type]
        return min(max(strength * (max_val - min_val) + min_val, min_val), max_val)

    def forward(self, image, strength=None):
        # Convert image to PIL image if it is a tensor
        if not isinstance(image, Image.Image):
            image = to_pil([image])[0]
        # Convert strength if relative
        if self.relative_strength:
            strength = self.relative_strength_to_absolute(strength)
        # Set the random seed for consistency
        set_random_seed(0 if self.same_operation else random.randint(0, 10000))
        # Get distortion parameters
        min_val, max_val = self.distortion_strength_params[self.distortion_type]
        strength = strength if strength is not None else random.uniform(min_val, max_val)
        # Apply the appropriate distortion
        distorted_image = self.apply_distortion(image, strength)
        # Convert to tensor if needed
        return to_tensor([distorted_image])[0] if not self.return_image else distorted_image

    def apply_distortion(self, image, strength):
        distortion_methods = {
            "rotation": self.apply_rotation,
            "resizedcrop": self.apply_resizedcrop,
            "erasing": self.apply_erasing,
            "brightness": self.apply_brightness,
            "contrast": self.apply_contrast,
            "blurring": self.apply_blurring,
            "noise": self.apply_noise,
            "compression": self.apply_compression,
        }
        if self.distortion_type not in distortion_methods:
            raise ValueError("Unsupported distortion type")
        return distortion_methods[self.distortion_type](image, strength)

    def apply_rotation(self, image, strength):
        return F.rotate(image, strength)

    def apply_resizedcrop(self, image, strength):
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(strength, strength), ratio=(1, 1))
        return F.resized_crop(image, i, j, h, w, image.size)

    def apply_erasing(self, image, strength):
        i, j, h, w, v = T.RandomErasing.get_params(to_tensor([image]), scale=(strength, strength), ratio=(1, 1), value=[0])
        return to_pil(F.erase(to_tensor([image], norm_type=None), i, j, h, w, v), norm_type=None)[0]

    def apply_brightness(self, image, strength):
        return ImageEnhance.Brightness(image).enhance(strength)

    def apply_contrast(self, image, strength):
        return ImageEnhance.Contrast(image).enhance(strength)

    def apply_blurring(self, image, strength):
        return image.filter(ImageFilter.GaussianBlur(int(strength)))

    def apply_noise(self, image, strength):
        noise = torch.randn(to_tensor([image]).size()) * strength
        return to_pil((to_tensor([image]) + noise).clamp(0, 1), norm_type=None)[0]

    def apply_compression(self, image, strength):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=int(strength))
        return Image.open(buffered)
