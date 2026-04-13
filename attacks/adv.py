import torch
import torchvision.transforms.functional as TF
from diffusers.models import AutoencoderKL
from torch import nn
from torchvision import models, transforms
from transformers import AutoProcessor, CLIPModel

# Constants for normalization
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
RESNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_INPUT_SIZE = (224, 224)
RESNET_INPUT_SIZE = [224, 224]
RESNET_LAYER_MAP = {
    "layer1": -6,
    "layer2": -5,
    "layer3": -4,
    "layer4": -3,
    "last": -1,
}


class BaseEncoder(nn.Module):
    def forward(self, images):
        raise NotImplementedError("This method should be implemented by subclasses.")


class ClipEmbedding(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.normalizer = transforms.Compose([
            transforms.Resize(CLIP_INPUT_SIZE),
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
    def __init__(self, layer: str):
        super().__init__()
        original_model = models.resnet18(pretrained=True)
        if layer not in RESNET_LAYER_MAP:
            raise ValueError("Invalid layer name")
        self.features = nn.Sequential(*list(original_model.children())[:RESNET_LAYER_MAP[layer]])

    def forward(self, images):
        images = TF.resize(images, RESNET_INPUT_SIZE)
        images = (images - RESNET_MEAN) / RESNET_STD
        return self.features(images)
