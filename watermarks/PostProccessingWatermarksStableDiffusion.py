from typing import List, Tuple

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from watermarks.DwtDct import DwtDCT
from watermarks.Rivagan import Rivagan
from watermarks.StegaStamp import StegaStamp

# Module-level constants
DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1"
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
RIVAGAN_DWT_MSG_LEN = 32
STEGASTAMP_MSG_LEN = 100


class PostProccessingWatermarksStableDiffusion:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        watermark_algorthim: str = 'rivagan',
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.pipe.requires_safety_checker = False
        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()

        if watermark_algorthim == 'rivagan':
            self.watermark_key = Rivagan()
        elif watermark_algorthim == 'dwtdct':
            self.watermark_key = DwtDCT(use_svd=False)
        elif watermark_algorthim == 'dwtdctsvd':
            self.watermark_key = DwtDCT(use_svd=True)
        elif watermark_algorthim == 'stegastamp':
            self.watermark_key = StegaStamp()

    def _default_messages(self, num_prompts: int) -> torch.Tensor:
        if isinstance(self.watermark_key, Rivagan) or isinstance(self.watermark_key, DwtDCT):
            return torch.randint(0, 2, (num_prompts, RIVAGAN_DWT_MSG_LEN)).float()
        if isinstance(self.watermark_key, StegaStamp):
            return torch.randint(0, 2, (num_prompts, STEGASTAMP_MSG_LEN)).float()
        return None

    def generate(
        self,
        prompts: List[str],
        watermark: bool = True,
        messages=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self.pipe(
            prompt=prompts,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            height=self.height,
            width=self.width,
        ).images
        if watermark:
            if messages is None:
                messages = self._default_messages(len(prompts))
            wm_images = self.watermark_key.encode(images, messages)
            return wm_images
        return images

    def decode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.watermark_key.decode(images)

    def generate_triplet(self, prompts, messages):
        images = self.pipe(
            prompt=prompts,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
        ).images
        if messages is None:
            messages = self._default_messages(len(prompts))
        inverse_messages = 1 - messages
        wm_images = self.watermark_key.encode(images, messages)
        inverse_wm_images = self.watermark_key.encode(images, inverse_messages)
        return images, wm_images, inverse_wm_images
