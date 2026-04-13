import numpy as np
import onnxruntime
import torch
from torchvision import transforms

# Module-level constants
MODEL_PATH = 'watermarks/stega_stamp.onnx'
ORT_PROVIDERS = [
    ('CUDAExecutionProvider', {'device_id': 0}),  # Specify the GPU device ID
    'CPUExecutionProvider',
]
DOWN_SIZE = (400, 400)
UP_SIZE = (512, 512)
MESSAGE_LENGTH = 100


class StegaStamp:
    def __init__(self):
        self.model = onnxruntime.InferenceSession(MODEL_PATH, providers=ORT_PROVIDERS)
        self.resize_down = transforms.Resize(DOWN_SIZE)
        self.resize_up = transforms.Resize(UP_SIZE)
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
