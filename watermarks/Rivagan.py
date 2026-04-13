import numpy as np
import onnxruntime
import torch
from torchvision import transforms

# Module-level constants
ENCODER_PATH = 'watermarks/rivagan_encoder.onnx'
DECODER_PATH = 'watermarks/rivagan_decoder.onnx'
ORT_PROVIDERS = [
    ('CUDAExecutionProvider', {'device_id': 0}),  # Specify the GPU device ID
    'CPUExecutionProvider',
]


class Rivagan:
    def __init__(self):
        self.encoder = onnxruntime.InferenceSession(ENCODER_PATH, providers=ORT_PROVIDERS)
        self.decoder = onnxruntime.InferenceSession(DECODER_PATH, providers=ORT_PROVIDERS)
        self._to_tensor = transforms.ToTensor()
        self._to_pil = transforms.ToPILImage()

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
