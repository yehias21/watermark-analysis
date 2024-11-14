import copy
import random
import numpy as np
import torch


class Trw:
    """
    The TRW watermarking key. Can be used to generate watermarked images and to extract the watermark.
    """
    def __init__(self,reversal_inference_steps = 20, channel=3, pattern='ring', mask_shape='circle', injection_type='complex',radius=10 , image_size=64,x_offset=0, y_offset=0):
        self.reversal_inference_steps = reversal_inference_steps
        self.channel = channel
        self.pattern = pattern
        self.mask_shape = mask_shape
        self.injection_type = injection_type
        self.image_size = image_size
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.radius = radius
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def sample_message(self, n: int) -> torch.Tensor:
        """ Sample n random message.
        """
        gt_init = torch.randn(*(n, 4, self.image_size, self.image_size),
                              device=self.device)

        if "random" in self.pattern:
            message = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            message[:] = message[0]
        elif "zeros" in self.pattern:
            message = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        elif "ring" in self.pattern:
            message = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch_tmp = copy.deepcopy(message)
            for i in range(self.radius, 0, -1):
                tmp_mask = self.circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(self.device)
                for j in range(message.shape[1]):
                    message[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
            return message.to(torch.complex64)  # 1, 4, img, img
        else:
            raise ValueError(self.pattern)
        return message

    def randomize_configuration(self):
        """ Sample random set of hyperparameters """
        self.x_offset = random.choice(list(range(-10, 10)))
        self.y_offset = random.choice(list(range(-10, 10)))
        self.radius = random.choice(list(range(7, 15)))
        self.channel = random.choice(list(range(0,4)))
        self.pattern = random.choice(["ring", "zeros", "random"])
        self.mask_shape = random.choice(["square", "circle"])
    
    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts an embedded_message from images.
        @param x should be in [n, c, h, w]
        """
        reversed_latents, _ = self.pipe.invert(x)
        fft_latents = torch.fft.fftshift(
            torch.fft.fft2(reversed_latents.to(self.device)), dim=(-1, -2)
        )
        return fft_latents  # this is the message [b, 4, 64, 64,]

    @torch.no_grad()
    def verify(self, msg_pred: torch.Tensor, msg_true: torch.Tensor) -> dict:
        """
        Extracts an embedded_message from images and computes a p-value for the confidence that the embedded
        and extracted messages are matching.
        """
        results = {'accuracy': 0, 'p_values': [0,0,0]}

        # non-watermarked messages:
        reversed_latents_no_w_fft = self.pipe.get_random_latents(batch_size=len(msg_pred))
        reversed_latents_no_w_fft = torch.fft.fftshift(
            torch.fft.fft2(reversed_latents_no_w_fft.to(self.device)), dim=(-1, -2)
        )
 
        watermarking_mask = self.get_watermarking_mask(reversed_latents_no_w_fft)

        no_w_metric, w_metric = [], []
        for i in range(reversed_latents_no_w_fft.size(0)):
            no_w_metric.append(
                torch.abs(
                    reversed_latents_no_w_fft[i][watermarking_mask[i]]
                    - msg_true[i][watermarking_mask[i]]
                )
                .mean()
                .item()
            )
        for i in range(msg_pred.size(0)):
            w_metric.append(
                torch.abs(
                    msg_pred[i][watermarking_mask[i]]
                    - msg_true[i][watermarking_mask[i]]
                )
                .mean()
                .item()
            )

        results['target'] = w_metric
        results['null'] = no_w_metric
        return results
    
    def set_message(self, msg: torch.Tensor):
        real_part = msg.real.float()
        imaginary_part = msg.imag.float()
        self.message = torch.stack([real_part, imaginary_part], 0)

    def get_message(self):
        return torch.complex(self.message[0], self.message[1]).to(
            self.device)

    def circle_mask(self, size, r=10):
        x0 = y0 = size // 2
        x0 += self.x_offset
        y0 += self.y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2

    def _inject_watermark(self, init_latents: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """ Injects the watermark into latents.
        """
        if message.size(0) != init_latents.size(0):
            message = message.repeat(len(init_latents), 1, 1, 1)

        watermarking_mask = self.get_watermarking_mask(init_latents)

        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents.to(torch.complex64)), dim=(-1, -2))
        if self.injection_type == 'complex':
            init_latents_fft[watermarking_mask] = message[watermarking_mask].clone()
        elif self.injection_type == 'seed':
            init_latents[watermarking_mask] = message.real[watermarking_mask].clone().half()
            return init_latents


        init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real
        return init_latents.half()

    def get_watermarking_mask(self, init_latents):
        watermarking_mask = torch.zeros(init_latents.shape, dtype=torch.bool).to(self.device)
        if self.mask_shape == "circle":
            np_mask = self.circle_mask(init_latents.shape[-1], r=self.radius)
            torch_mask = torch.tensor(np_mask).to(self.device)

            if self.channel == -1:
                # all channels
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, self.channel] = torch_mask
        elif self.mask_shape == "square":
            anchor_p = init_latents.shape[-1] // 2
            if self.channel == -1:
                # all channels
                watermarking_mask[
                :,
                :,
                anchor_p - self.radius: anchor_p + self.radius,
                anchor_p - self.radius: anchor_p + self.radius,
                ] = True
            else:
                watermarking_mask[
                :,
                self.channel,
                anchor_p - self.radius: anchor_p + self.radius,
                anchor_p - self.radius: anchor_p + self.radius,
                ] = True
            
        return watermarking_mask
