import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision import transforms
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018,
    cheng2020_anchor,
)
import numpy as np
import argparse
from diffusers import DiffusionPipeline
from abc import ABC, abstractmethod
from PIL import Image
import os
from copy import deepcopy
from glob import glob
from PIL import Image
os.environ["TORCH_HOME"] = "~/.cache_new"

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)

class generativeAttacks(ABC):
    @abstractmethod
    def attack(self, image):
        pass

class VAEAttack(generativeAttacks):
    def __init__(self, model=[], quality=1, metric="mse"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = []
        self.model_names = []
        for m in model:
            print(f"Loading {m} model")
            match m:
                case "bmshj2018_factorized":
                    self.model.append(bmshj2018_factorized(quality=quality, metric=metric, pretrained=True).eval().to(self.device))
                    self.model_names.append(m)
                case "bmshj2018_hyperprior":
                    self.model.append(bmshj2018_hyperprior(quality=quality, metric=metric, pretrained=True).eval().to(self.device))
                    self.model_names.append(m)
                case "mbt2018_mean":
                    self.model.append(mbt2018_mean(quality=quality, metric=metric, pretrained=True).eval().to(self.device))
                    self.model_names.append(m)
                case "mbt2018":
                    self.model.append(mbt2018(quality=quality, metric=metric, pretrained=True).eval().to(self.device))
                    self.model_names.append(m)
                case "cheng2020_anchor":
                    self.model.append(cheng2020_anchor(quality=quality, metric=metric, pretrained=True).eval().to(self.device))
                    self.model_names.append(m)
                case _:
                    raise Exception("Invalid model name")

    def attack(self, dataloader, output_dir, preprocess=True):
        os.makedirs(output_dir, exist_ok=True)
        for model, model_name in zip(self.model, self.model_names):
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            for images, image_names in dataloader:
                images = images.to(self.device)
                with torch.no_grad():
                    out = model(images)
                    out["x_hat"].clamp_(0, 1)
                    results = out["x_hat"].cpu()
                for i, image_name in enumerate(image_names):
                    output_image = transforms.ToPILImage()(results[i])
                    output_path = os.path.join(model_output_dir, image_name)
                    output_image.save(output_path)

class DiffuserAttack:
    def __init__(self, model="stabilityai/stable-diffusion-xl-refiner-1.0"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipeline = DiffusionPipeline.from_pretrained(model
    , torch_dtype=torch.float16, variant="fp16", use_safetensors=True, 
).to(self.device)

    def attack(self, images_path,out, regen=1, prompt=""):
        images = [Image.open(img).convert("RGB") for img in images_path]
        out = os.path.join(out, "diffuser")
        os.makedirs(out, exist_ok=True)
        for idx,img in enumerate(images):
            for iter in range(regen):
                img = self.pipeline(prompt, image=img).images[0]
            img.save(out+"/"+str(os.path.basename(images_path[idx])))


# Example usage
image_dir = "/ephemeral/yaya/projects/watermark-analysis/data/Neurips24_ETI_BlackBox"
output_dir = "/ephemeral/yaya/projects/watermark-analysis/data/outout"
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
dataset = ImageDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Initialize attack
# vae_attack = VAEAttack(model=["bmshj2018_factorized", "cheng2020_anchor"], quality=1, metric="mse")
# vae_attack.attack(dataloader, output_dir)

diffuser_attack = DiffuserAttack(model="stabilityai/stable-diffusion-xl-refiner-1.0")
diffuser_attack.attack(glob(image_dir+"/*.png"), output_dir, regen=1, prompt="")