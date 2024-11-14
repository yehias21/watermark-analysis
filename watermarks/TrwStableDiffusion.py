from typing import Tuple, List

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler
from watermarks.ModifiedStableDiffusionPipeline import ModifiedStableDiffusionPipeline
from watermarks.Trw import Trw

class TrwStableDiffusion:

    def __init__(self, model=  "stabilityai/stable-diffusion-2",num_inference_steps = 20,guidance_scale= 7.5, channel=3, pattern='ring', mask_skape='circle', injection_type='complex', image_size=64,x_offset=0, y_offset=0):
        self.pipe  = ModifiedStableDiffusionPipeline.from_pretrained(
            model,
            scheduler=DDIMScheduler.from_pretrained(model, subfolder='scheduler'),
            torch_dtype=torch.float16,
            variant="fp16"
        )

        self.pipe.inverse_scheduler = DDIMInverseScheduler.from_pretrained(model, subfolder='scheduler')
        self.pipe.requires_safety_checker = False
        self.pipe.set_progress_bar_config(disable=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trw = Trw(num_inference_steps, channel, pattern, mask_skape, injection_type, image_size, x_offset, y_offset)
        self.pipe.to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def generate(self,
                 prompts: List[str],
                 watermark= True,
                 messages= None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate images using TRW. Optionally inject a watermark if a key was defined.
        """
        latents =  self.pipe.get_random_latents(batch_size=len(prompts)).to(self.device)
        if watermark:
            if messages is None: 
                messages =  torch.repeat_interleave(self.wm_key.get_message().unsqueeze(0), len(prompts), dim=0)
            self.trw.set_message(messages)
            latents= self.trw._inject_watermark(latents, messages)
        return self.pipe(prompt=prompts, latents=latents,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps).images


    def generate_image_latent_pair(self, prompts: List[str], watermark= True, messages= None):
        latents =  self.pipe.get_random_latents(batch_size=len(prompts)).to(self.device)
        if watermark:
            if messages is None: 
                messages =  self.trw.sample_message(len(prompts))
            self.trw.set_message(messages)
            latents= self.trw._inject_watermark(latents, messages)
        return self.pipe(prompt=prompts, latents=latents,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps, return_latents=True)
        


    def generate_triplet(self, prompts, messages):
        """
        Generate images and their watermarked and inverse watermarked versions.
        """
        if messages is None:
            messages = self.trw.sample_message(len(prompts))
        inverse_messages = 1 - messages
        latents =  self.pipe.get_random_latents(batch_size=len(prompts)).to(self.device)
        wm_latents= self.trw._inject_watermark(latents, messages)
        inverse_wm_latents = self.trw._inject_watermark(latents, inverse_messages)
        images = self.pipe(prompt=prompts, latents=latents, guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps).images
        wm_images = self.pipe(prompt=prompts, latents=wm_latents,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps)
        inverse_wm_images = self.pipe(prompt=prompts, latents=inverse_wm_latents,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps)
        return images, wm_images, inverse_wm_images

    def invert(self, x, *args, **kwargs) -> torch.Tensor:
        return self.pipe.invert(x, *args, **kwargs)

    def get_random_latents(self, *args, **kwargs):
        """
        Generate images.
        """
        return self.pipe.get_random_latents(*args, **kwargs).to(torch.complex64)
