from typing import Tuple, List
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from watermarks.Rivagan import RivaGan
from watermarks.StegaStamp import StegaStamp

class PostProccessingWatermarksStableDiffusion:

    def __init__(self, model=  "stabilityai/stable-diffusion-2-1",watermark_algorthim='rivagan', num_inference_steps = 20, guidance_scale= 7.5):
        self.pipe  = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.pipe.requires_safety_checker = False
        self.pipe.set_progress_bar_config(disable=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)
        torch.cuda.manual_seed_all(333)
        
        if watermark_algorthim == 'rivagan':
            self.watermark_key = RivaGan()
        elif watermark_algorthim == 'stegastamp':
            self.watermark_key = StegaStamp()

    def generate(self,
                 prompts: List[str],
                 watermark= True,
                  messages= None) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self.pipe(prompt=prompts,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps).images
        if watermark:
            if messages is None:
                if isinstance(self.watermark_key, RivaGan):
                    messages = torch.randint(0, 2, (len(prompts), 32)).float()
                elif isinstance(self.watermark_key, StegaStamp):
                    messages = torch.randint(0, 2, (len(prompts), 100)).float()
            wm_images = self.watermark_key.encode(images,messages)
            return wm_images
        return images
    
    def decode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.watermark_key.decode(images)
    
    def generate_triplet(self,
                 prompts,
                 messages):
        images = self.pipe(prompt=prompts,guidance_scale = self.guidance_scale, num_inference_steps=self.num_inference_steps).images
        if messages is None:
            if isinstance(self.watermark_key, RivaGan):
                messages = torch.randint(0, 2, (len(prompts), 32)).float()
            elif isinstance(self.watermark_key, StegaStamp):
                messages = torch.randint(0, 2, (len(prompts), 100)).float()
        inverse_messages = 1 - messages
        wm_images = self.watermark_key.encode(images,messages)
        inverse_wm_images = self.watermark_key.encode(images,inverse_messages)
        return images, wm_images, inverse_wm_images
