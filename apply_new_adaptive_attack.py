import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Autoencode images using a fine-tuned VAE.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to save reconstructed images.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to the fine-tuned VAE model.")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the fine-tuned VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="vae", torch_dtype=torch.float32)
    vae.load_state_dict(torch.load(args.vae_path))
    vae.to(device)
    vae.eval()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Define image preprocessing and postprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    postprocess = transforms.Compose([
        transforms.Normalize([-1], [2]),  # Convert from [-1, 1] to [0, 1]
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage()
    ])

    # Get list of image files in the input folder
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    image_filenames = [f for f in os.listdir(args.input_folder) if os.path.splitext(f)[1].lower() in supported_extensions]

    # Process each image
    for filename in tqdm(image_filenames, desc="Processing images"):
        input_path = os.path.join(args.input_folder, filename)
        output_path = os.path.join(args.output_folder, filename)

        # Load and preprocess the image
        image = Image.open(input_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Encode and decode the image
            latents = vae.encode(image_tensor).latent_dist.sample()
            reconstructed = vae.decode(latents).sample

        # Postprocess and save the reconstructed image
        reconstructed_image = postprocess(reconstructed.squeeze(0).cpu())
        reconstructed_image.save(output_path)

if __name__ == "__main__":
    main()