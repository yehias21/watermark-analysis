import torch
from torch import nn
from transformers import AutoProcessor, CLIPModel
from torchvision import transforms, models
from diffusers.models import AutoencoderKL
import torchvision.transforms.functional as TF

# Constants for normalization
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
RESNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


class BaseEncoder(nn.Module):
    def forward(self, images):
        raise NotImplementedError("This method should be implemented by subclasses.")


class ClipEmbedding(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.normalizer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        ])

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x).cuda())
        return self.model.get_image_features(**inputs)


class VAEEmbedding(BaseEncoder):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_name)

    def forward(self, images):
        images = 2.0 * images - 1.0
        return self.model.encode(images).latent_dist.mode()


class ResNet18Embedding(BaseEncoder):
    def __init__(self, layer):
        super().__init__()
        original_model = models.resnet18(pretrained=True)
        layer_map = {
            "layer1": -6,
            "layer2": -5,
            "layer3": -4,
            "layer4": -3,
            "last": -1
        }
        if layer not in layer_map:
            raise ValueError("Invalid layer name")
        self.features = nn.Sequential(*list(original_model.children())[:layer_map[layer]])

    def forward(self, images):
        images = TF.resize(images, [224, 224])
        images = (images - RESNET_MEAN) / RESNET_STD
        return self.features(images)
