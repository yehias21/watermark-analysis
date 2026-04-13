import argparse
import os
from typing import List

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Module-level constants
VAE_PRETRAINED_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
IMAGE_SIZE = (512, 512)
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoencode images using a fine-tuned VAE.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to save reconstructed images.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to the fine-tuned VAE model.")
    return parser.parse_args()


def load_vae(vae_path: str, device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(VAE_PRETRAINED_MODEL, subfolder="vae", torch_dtype=torch.float32)
    vae.load_state_dict(torch.load(vae_path))
    vae.to(device)
    vae.eval()
    return vae


def build_transforms():
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ])
    postprocess = transforms.Compose([
        transforms.Normalize([-1], [2]),  # Convert from [-1, 1] to [0, 1]
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage(),
    ])
    return preprocess, postprocess


def list_image_files(folder: str) -> List[str]:
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = load_vae(args.vae_path, device)

    os.makedirs(args.output_folder, exist_ok=True)

    preprocess, postprocess = build_transforms()

    image_filenames = list_image_files(args.input_folder)

    for filename in tqdm(image_filenames, desc="Processing images"):
        input_path = os.path.join(args.input_folder, filename)
        output_path = os.path.join(args.output_folder, filename)

        image = Image.open(input_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            latents = vae.encode(image_tensor).latent_dist.sample()
            reconstructed = vae.decode(latents).sample

        reconstructed_image = postprocess(reconstructed.squeeze(0).cpu())
        reconstructed_image.save(output_path)


if __name__ == "__main__":
    main()
