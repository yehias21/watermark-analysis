import onnxruntime
import numpy as np
import torch
from torchvision import transforms


class Rivagan:
    def __init__(self):
        providers = [
    ('CUDAExecutionProvider', {'device_id': 5}),  # Specify the GPU device ID
    'CPUExecutionProvider']
        self.encoder =  onnxruntime.InferenceSession('watermarks/rivagan_encoder.onnx', providers=providers)
        self.decoder =  onnxruntime.InferenceSession('watermarks/rivagan_decoder.onnx', providers=providers)

    def encode(self, images, messages):
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        images = (images - 0.5) * 2
        images = torch.clamp(images,-1.0, 1.0)
        images = images.unsqueeze(2)
        inputs = {
            'frame': images.detach().cpu().numpy(),
            'data': messages.detach().cpu().numpy(),
        }
        wm_images = np.stack(self.encoder.run(None, inputs))
        wm_images = torch.clamp(torch.from_numpy(wm_images), min=-1.0, max=1.0)
        wm_images = (wm_images / 2) + 0.5
        wm_images = wm_images.squeeze()
        to_pil = transforms.ToPILImage()
        return [to_pil(wm_images[i])  for i in range(wm_images.size(0))]

    def decode(self, images):
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        images = (images - 0.5) * 2
        images = torch.clamp(images,-1.0, 1.0)
        images = images.unsqueeze(2)
        inputs = {
            'frame': images.detach().cpu().numpy(),
        }
        outputs = self.decoder.run(None, inputs)
        messages = outputs[0]
        messages = (messages > 0.5).astype(np.uint8)
        return messages