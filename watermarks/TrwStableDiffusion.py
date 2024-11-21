from typing import Tuple, List

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler
from watermarks.ModifiedStableDiffusionPipeline import ModifiedStableDiffusionPipeline
from watermarks.Trw import Trw
from torchvision import transforms

class TrwStableDiffusion:

    def __init__(self, model=  "stabilityai/stable-diffusion-2",num_inference_steps = 20,guidance_scale= 7.5, channel=3, height=512, width=512, pattern='ring', mask_shape='circle', injection_type='complex', radius=10, image_size=64, x_offset=0, y_offset=0):
        self.pipe  = ModifiedStableDiffusionPipeline.from_pretrained(
            model,
            scheduler=DDIMScheduler.from_pretrained(model, subfolder='scheduler'),
            torch_dtype=torch.float16,
            variant="fp16",
        )

        self.pipe.inverse_scheduler = DDIMInverseScheduler.from_pretrained(model, subfolder='scheduler')
        self.pipe.requires_safety_checker = False
        self.pipe.set_progress_bar_config(disable=True)
        self.device =  "cuda:7"# torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trw = Trw(reversal_inference_steps=num_inference_steps, channel=channel, pattern= pattern, mask_shape=mask_shape, injection_type= injection_type, radius=radius, image_size= image_size, x_offset= x_offset, y_offset= y_offset)
        self.pipe.to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width

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
                self.trw.set_message()
            self.trw.set_message(messages)
            messages =  self.trw.get_message().to(self.device)
            latents= self.trw._inject_watermark(latents, messages)
        images =self.pipe(prompt=prompts, latents=latents,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps, height=self.height, width=self.width)[1]
        to_pil_image = transforms.ToPILImage()
        images = [to_pil_image(image) for image in images]
        return images

    def generate_image_latent_pair(self, prompts: List[str], watermark= True, messages= None):
        latents =  self.pipe.get_random_latents(batch_size=len(prompts)).to(self.device)
        if watermark:
            if messages is None: 
                self.trw.set_message(self.trw.sample_message(1)[0])
                messages = torch.repeat_interleave(
                self.trw.get_message().to(self.device).unsqueeze(0), 
                len(prompts), 
                dim=0
            )
            latents= self.trw._inject_watermark(latents, messages)
        return self.pipe(prompt=prompts, latents=latents,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps, return_latents=True)
        


    def generate_triplet(self, prompts, messages):
        """
        Generate images and their watermarked and inverse watermarked versions.
        """
        if messages is None:
            messages = self.trw.sample_message(len(prompts))
        inverse_messages =   messages * -1
        latents =  self.pipe.get_random_latents(batch_size=len(prompts)).to(self.device)
        wm_latents= self.trw._inject_watermark(latents, messages)
        inverse_wm_latents = self.trw._inject_watermark(latents, inverse_messages)
        images = self.pipe(prompt=prompts, latents=latents, guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps)[1]
        wm_images = self.pipe(prompt=prompts, latents=wm_latents,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps)[1]
        inverse_wm_images = self.pipe(prompt=prompts, latents=inverse_wm_latents,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps)[1]
        to_pil_image = transforms.ToPILImage()
        images = [to_pil_image(image) for image in images]
        wm_images = [to_pil_image(image) for image in wm_images]
        inverse_wm_images = [to_pil_image(image) for image in inverse_wm_images]
        return images, wm_images, inverse_wm_images

    def invert(self, x, *args, **kwargs) -> torch.Tensor:
        return self.pipe.invert(x, *args, **kwargs)

    def get_random_latents(self, *args, **kwargs):
        """
        Generate images.
        """
        return self.pipe.get_random_latents(*args, **kwargs).to(torch.complex64)
