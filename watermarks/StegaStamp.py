import onnxruntime
import numpy as np
import torch
from torchvision import transforms


class StegaStamp:
    def __init__(self):
        self.model =  onnxruntime.InferenceSession('watermarks/stega_stamp.onnx')
        self.resize_down = transforms.Resize((400, 400))
        self.resize_up = transforms.Resize((512, 512))
        
    def encode(self, images, messages):
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        inputs = {
            'image': self.resize_down(images).permute(0, 2, 3, 1).detach().cpu().float().numpy(),
            'secret': messages.detach().cpu().numpy(),
        }
        wm_images = np.stack(self.model.run(None, inputs)[0])
        wm_images = torch.from_numpy(wm_images)
        wm_images = wm_images.permute(0, 3, 1, 2)
        wm_images = self.resize_up(wm_images)
        to_pil = transforms.ToPILImage()
        return [to_pil(wm_images[i])  for i in range(wm_images.size(0))]

    def decode(self, images):
        images = torch.stack([transforms.ToTensor(img) for img in images])
        images = images.unsqueeze(2)
        inputs = {
            'image': self.resize_down(images).permute(0, 2, 3, 1).detach().cpu().numpy()
        }
        outputs = self.decoder.run(None, inputs)
        messages = outputs[0]
        messages = (messages > 0.5).astype(np.uint8)
        return messages