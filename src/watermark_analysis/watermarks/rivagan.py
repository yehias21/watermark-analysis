import numpy as np
import onnxruntime
import torch
from torchvision import transforms

from ..config import ORT_PROVIDERS, RIVAGAN_DECODER_PATH, RIVAGAN_ENCODER_PATH
from .base import Watermark


class Rivagan(Watermark):
    def __init__(self):
        self.encoder = onnxruntime.InferenceSession(RIVAGAN_ENCODER_PATH, providers=ORT_PROVIDERS)
        self.decoder = onnxruntime.InferenceSession(RIVAGAN_DECODER_PATH, providers=ORT_PROVIDERS)
        self._to_tensor = transforms.ToTensor()
        self._to_pil = transforms.ToPILImage()

    # Watermark ABC -------------------------------------------------------
    def embed(self, image, message):
        msg = message if torch.is_tensor(message) else torch.as_tensor(message)
        return self.encode([image], msg.unsqueeze(0) if msg.dim() == 1 else msg)[0]

    def extract(self, image):
        return self.decode([image])[0]

    def _prepare_frames(self, images):
        images = torch.stack([self._to_tensor(img) for img in images])
        images = (images - 0.5) * 2
        images = torch.clamp(images, -1.0, 1.0)
        return images.unsqueeze(2)

    def encode(self, images, messages):
        frames = self._prepare_frames(images)
        inputs = {
            'frame': frames.detach().cpu().numpy(),
            'data': messages.detach().cpu().numpy(),
        }
        wm_images = np.stack(self.encoder.run(None, inputs))
        wm_images = torch.clamp(torch.from_numpy(wm_images), min=-1.0, max=1.0)
        wm_images = (wm_images / 2) + 0.5
        wm_images = wm_images.squeeze()
        return [self._to_pil(wm_images[i]) for i in range(wm_images.size(0))]

    def decode(self, images):
        frames = self._prepare_frames(images)
        inputs = {
            'frame': frames.detach().cpu().numpy(),
        }
        outputs = self.decoder.run(None, inputs)
        messages = outputs[0]
        return messages
