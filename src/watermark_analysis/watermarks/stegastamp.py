import numpy as np
import onnxruntime
import torch
from torchvision import transforms

from ..config import (
    ORT_PROVIDERS,
    STEGASTAMP_DOWN_SIZE as DOWN_SIZE,
    STEGASTAMP_MODEL_PATH,
    STEGASTAMP_MSG_LEN as MESSAGE_LENGTH,
    STEGASTAMP_UP_SIZE as UP_SIZE,
)
from .base import Watermark


class StegaStamp(Watermark):
    def __init__(self):
        self.model = onnxruntime.InferenceSession(STEGASTAMP_MODEL_PATH, providers=ORT_PROVIDERS)
        self.resize_down = transforms.Resize(DOWN_SIZE)
        self.resize_up = transforms.Resize(UP_SIZE)

    # Watermark ABC -------------------------------------------------------
    def embed(self, image, message):
        msg = message if torch.is_tensor(message) else torch.as_tensor(message)
        return self.encode([image], msg.unsqueeze(0) if msg.dim() == 1 else msg)[0]

    def extract(self, image):
        return self.decode([image])[0]
        self._to_tensor = transforms.ToTensor()
        self._to_pil = transforms.ToPILImage()

    def encode(self, images, messages):
        images = torch.stack([self._to_tensor(img) for img in images])
        inputs = {
            'image': self.resize_down(images).permute(0, 2, 3, 1).detach().cpu().float().numpy(),
            'secret': messages.detach().cpu().numpy(),
        }
        wm_images = np.stack(self.model.run(None, inputs)[0])
        wm_images = torch.from_numpy(wm_images)
        wm_images = wm_images.permute(0, 3, 1, 2)
        wm_images = self.resize_up(wm_images)
        return [self._to_pil(wm_images[i]) for i in range(wm_images.size(0))]

    def decode(self, images):
        images = torch.stack([self._to_tensor(img) for img in images])
        inputs = {
            'image': self.resize_down(images).permute(0, 2, 3, 1).detach().cpu().float().numpy(),
            "secret": np.zeros((len(images), MESSAGE_LENGTH), dtype=np.float32),
        }
        outputs = self.model.run(None, inputs)
        messages = outputs[2]
        return messages
