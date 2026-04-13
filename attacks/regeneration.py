import os
from abc import ABC, abstractmethod
from glob import glob

import torch
import torchvision.transforms as transforms
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    mbt2018,
    mbt2018_mean,
)
from diffusers import DiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ["TORCH_HOME"] = "~/.cache_new"

# Module-level constants
DEFAULT_DIFFUSER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
IMAGE_SIZE = (512, 512)
VALID_IMAGE_EXTENSIONS = ('png', 'jpg', 'jpeg')
COMPRESSAI_MODEL_BUILDERS = {
    "bmshj2018_factorized": bmshj2018_factorized,
    "bmshj2018_hyperprior": bmshj2018_hyperprior,
    "mbt2018_mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020_anchor": cheng2020_anchor,
}


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith(VALID_IMAGE_EXTENSIONS)
        ]
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = []
        self.model_names = []
        for m in model:
            print(f"Loading {m} model")
            if m not in COMPRESSAI_MODEL_BUILDERS:
                raise Exception("Invalid model name")
            builder = COMPRESSAI_MODEL_BUILDERS[m]
            self.model.append(builder(quality=quality, metric=metric, pretrained=True).eval().to(self.device))
            self.model_names.append(m)

    def attack(self, dataloader, output_dir, preprocess=True):
        to_pil = transforms.ToPILImage()
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
                    output_image = to_pil(results[i])
                    output_path = os.path.join(model_output_dir, image_name)
                    output_image.save(output_path)


class DiffuserAttack:
    def __init__(self, model: str = DEFAULT_DIFFUSER_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(self.device)

    def attack(self, images_path, out, regen=1, prompt=""):
        images = [Image.open(img).convert("RGB") for img in images_path]
        out = os.path.join(out, "diffuser")
        os.makedirs(out, exist_ok=True)
        for idx, img in enumerate(images):
            for _ in range(regen):
                img = self.pipeline(prompt, image=img).images[0]
            img.save(out + "/" + str(os.path.basename(images_path[idx])))


# Example usage
image_dir = "/ephemeral/tbakr/watermark-analysis/cache/test_dataset_stegastamp"
output_dir = "/ephemeral/tbakr/watermark-analysis/attacked/test_dataset_stegastamp"
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])
dataset = ImageDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Initialize attack
vae_attack = VAEAttack(
    model=["bmshj2018_factorized", "cheng2020_anchor", "mbt2018", "bmshj2018_hyperprior", "mbt2018_mean"],
    quality=1,
    metric="mse",
)
vae_attack.attack(dataloader, output_dir)

diffuser_attack = DiffuserAttack(model=DEFAULT_DIFFUSER_MODEL)
diffuser_attack.attack(glob(image_dir + "/*.png"), output_dir, regen=2, prompt="")
