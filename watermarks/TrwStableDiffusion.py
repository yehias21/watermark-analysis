from typing import List, Tuple

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler
from torchvision import transforms

from watermarks.ModifiedStableDiffusionPipeline import ModifiedStableDiffusionPipeline
from watermarks.Trw import Trw

# Module-level constants
DEFAULT_MODEL = "stabilityai/stable-diffusion-2"
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CHANNEL = 3
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_PATTERN = 'ring'
DEFAULT_MASK_SHAPE = 'circle'
DEFAULT_INJECTION_TYPE = 'complex'
DEFAULT_RADIUS = 10
DEFAULT_IMAGE_SIZE = 64
DEVICE_OVERRIDE = "cuda:7"


class TrwStableDiffusion:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        channel: int = DEFAULT_CHANNEL,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        pattern: str = DEFAULT_PATTERN,
        mask_shape: str = DEFAULT_MASK_SHAPE,
        injection_type: str = DEFAULT_INJECTION_TYPE,
        radius: int = DEFAULT_RADIUS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        x_offset: int = 0,
        y_offset: int = 0,
    ):
        self.pipe = ModifiedStableDiffusionPipeline.from_pretrained(
            model,
            scheduler=DDIMScheduler.from_pretrained(model, subfolder='scheduler'),
            torch_dtype=torch.float16,
            variant="fp16",
        )

        self.pipe.inverse_scheduler = DDIMInverseScheduler.from_pretrained(model, subfolder='scheduler')
        self.pipe.requires_safety_checker = False
        self.pipe.set_progress_bar_config(disable=True)
        self.device = DEVICE_OVERRIDE  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trw = Trw(
            reversal_inference_steps=num_inference_steps,
            channel=channel,
            pattern=pattern,
            mask_shape=mask_shape,
            injection_type=injection_type,
            radius=radius,
            image_size=image_size,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        self.pipe.to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self._to_pil_image = transforms.ToPILImage()

    def generate(
        self,
        prompts: List[str],
        watermark: bool = True,
        messages=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate images using TRW. Optionally inject a watermark if a key was defined."""
        latents = self.pipe.get_random_latents(batch_size=len(prompts)).to(self.device)
        if watermark:
            if messages is None:
                self.trw.set_message()
            self.trw.set_message(messages)
            messages = self.trw.get_message().to(self.device)
            latents = self.trw._inject_watermark(latents, messages)
        images = self.pipe(
            prompt=prompts,
            latents=latents,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            height=self.height,
            width=self.width,
        )[1]
        return [self._to_pil_image(image) for image in images]

    def generate_image_latent_pair(self, prompts: List[str], watermark: bool = True, messages=None):
        latents = self.pipe.get_random_latents(batch_size=len(prompts)).to(self.device)
        if watermark:
            if messages is None:
                self.trw.set_message(self.trw.sample_message(1)[0])
                messages = torch.repeat_interleave(
                    self.trw.get_message().to(self.device).unsqueeze(0),
                    len(prompts),
                    dim=0,
                )
            latents = self.trw._inject_watermark(latents, messages)
        return self.pipe(
            prompt=prompts,
            latents=latents,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            return_latents=True,
        )

    def generate_triplet(self, prompts, messages):
        """Generate images and their watermarked and inverse watermarked versions."""
        if messages is None:
            messages = self.trw.sample_message(len(prompts))
        inverse_messages = messages * -1
        latents = self.pipe.get_random_latents(batch_size=len(prompts)).to(self.device)
        wm_latents = self.trw._inject_watermark(latents, messages)
        inverse_wm_latents = self.trw._inject_watermark(latents, inverse_messages)
        images = self.pipe(prompt=prompts, latents=latents, guidance_scale=self.guidance_scale, num_inference_steps=self.num_inference_steps)[1]
        wm_images = self.pipe(prompt=prompts, latents=wm_latents, guidance_scale=self.guidance_scale, num_inference_steps=self.num_inference_steps)[1]
        inverse_wm_images = self.pipe(prompt=prompts, latents=inverse_wm_latents, guidance_scale=self.guidance_scale, num_inference_steps=self.num_inference_steps)[1]
        images = [self._to_pil_image(image) for image in images]
        wm_images = [self._to_pil_image(image) for image in wm_images]
        inverse_wm_images = [self._to_pil_image(image) for image in inverse_wm_images]
        return images, wm_images, inverse_wm_images

    def invert(self, x, *args, **kwargs) -> torch.Tensor:
        return self.pipe.invert(x, *args, **kwargs)

    def get_random_latents(self, *args, **kwargs):
        """Generate images."""
        return self.pipe.get_random_latents(*args, **kwargs).to(torch.complex64)
