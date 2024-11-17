import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
import torch
import io
import argparse

class DistortionAttacks():
    def __init__(self):
        super(DistortionAttacks, self).__init__()
        
        self.distortion_strength_params = {
            "rotation": 25,
            "resizedcrop": .5,
            "erasing": 0.25,
            "brightness": .6,
            "contrast": 0.8,
            "blurring": 7,
            "noise":  0.1,
            "compression": 80,
        }
        self.distortion_methods = {
            "rotation": self.apply_rotation,
            "resizedcrop": self.apply_resizedcrop,
            "erasing": self.apply_erasing,
            "brightness": self.apply_brightness,
            "contrast": self.apply_contrast,
            "blurring": self.apply_blurring,
            "noise": self.apply_noise,
            "compression": self.apply_compression,
        }

    def process_single_image(self, image, distortion_type):
        return  self.distortion_methods[distortion_type](image, self.distortion_strength_params[distortion_type])

    def apply_rotation(self, image, params = 25):
        return  transforms.functional.rotate(image, params)

    def apply_resizedcrop(self, image, params=0.5):
        image = transforms.ToTensor()(image)
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(params,params), ratio=(1, 1))
        image = transforms.functional.resized_crop(image, i, j, h, w, image.size()[1:])
        image = transforms.ToPILImage()(image)
        return image

    def apply_erasing(self, image, params = 0.25):
        image = transforms.ToTensor()(image)
        i, j, h, w, v = transforms.RandomErasing.get_params(image, scale=(params,params), ratio=(1, 1), value=None)
        transforms.functional.erase(image, i, j, h, w, v)
        image = transforms.ToPILImage()(image)
        return image
    

    def apply_brightness(self, image, params = 1):
        return ImageEnhance.Brightness(image).enhance(params)

    def apply_contrast(self, image, params = 0.8):
        return ImageEnhance.Contrast(image).enhance(params)

    def apply_blurring(self, image, params = 7):
        return image.filter(ImageFilter.GaussianBlur(params))

    def apply_noise(self, image, params = 0.1):
        image = transforms.ToTensor()(image)
        noise = torch.randn_like(image) * params
        image = (image + noise).clamp(0, 1)
        image = transforms.ToPILImage()(image)
        return image


    def apply_compression(self, image, params = 80):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=params)
        return Image.open(buffered)

    def process_directory(self, input_dir, output_dir):
        input_dirname = os.path.basename(os.path.normpath(input_dir))
        image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for ext in valid_extensions:
            image_files.extend(list(Path(input_dir).glob(f'*{ext}')))

        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            # Load image
            img = Image.open(img_path)
            img_name = img_path.stem
            
            # Process with each distortion type
            for dist_type in self.distortion_strength_params.keys():
                # Create output directory with format: /output_dir/input_dirname_transformation_name
                dist_dir = os.path.join(output_dir, f"{input_dirname}_{dist_type}")
                os.makedirs(dist_dir, exist_ok=True)
                
                # Apply distortion
                distorted = self.process_single_image(img, dist_type)
                
                # Save distorted image
                save_path = os.path.join(dist_dir, f"{img_name}.png")
                distorted.save(save_path)


def main():
    parser = argparse.ArgumentParser(description='Apply image distortions')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for distorted images')
    
    args = parser.parse_args()
    
    processor = DistortionAttacks()
    processor.process_directory(args.input_dir, args.output_dir)
    print("Processing complete!")

if __name__ == "__main__":
    main()