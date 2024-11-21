from prompts_generator import next_prompt
import tqdm
import os
from watermarks.PostProccessingWatermarksStableDiffusion import PostProccessingWatermarksStableDiffusion
from watermarks.TrwStableDiffusion import TrwStableDiffusion
import torch
import pandas as pd

@torch.no_grad()
def create_attack_dataset(number_of_samples: int, batch_size: int, cache_dir: str,watermark_algorthim='rivagan'):
    prompt_iterator = next_prompt()
    cache_dir_no_watermark = os.path.join(cache_dir, f'attack_dataset_{watermark_algorthim}', 'no_watermark')
    cache_dir_watermark = os.path.join(cache_dir, f'attack_dataset_{watermark_algorthim}', 'watermark')
    cache_dir_inverse_watermark = os.path.join(cache_dir, f'attack_dataset_{watermark_algorthim}', 'inverse_watermark')
    os.makedirs(cache_dir_no_watermark, exist_ok=True)
    os.makedirs(cache_dir_watermark, exist_ok=True)
    os.makedirs(cache_dir_inverse_watermark, exist_ok=True)
    if watermark_algorthim == 'trw':
        generator = TrwStableDiffusion()
    else:
        generator = PostProccessingWatermarksStableDiffusion(model= "sd-legacy/stable-diffusion-v1-5", watermark_algorthim=watermark_algorthim)
    tqdm_bar = tqdm.tqdm(range(number_of_samples // batch_size), desc="Generating Attack Dataset")
    data_acc = []
    image_index = 0

    for batch in tqdm_bar:
        prompts = [next(prompt_iterator) for _ in range(batch_size)]
        if watermark_algorthim == 'trw':
            generator.trw.set_message(generator.trw.sample_message(1)[0])
            messages = torch.repeat_interleave(
            generator.trw.get_message().to(generator.device).unsqueeze(0), 
            len(prompts), 
            dim=0
        )
        elif watermark_algorthim == 'rivagan' or watermark_algorthim == 'dwtdct' or watermark_algorthim == 'dwtdctsvd':
            messages = torch.randint(0, 2, (len(prompts), 32)).float()
        elif watermark_algorthim == 'stegastamp':
            messages = torch.randint(0, 2, (len(prompts), 100)).float()
        images_no_watermarked, images_watermarked, images_inverse_watermarked = generator.generate_triplet(prompts, messages)

        for i, (image_no_watermark, image_watermark, image_inverse_watermark, prompt, message) in enumerate(
                zip(images_no_watermarked, images_watermarked, images_inverse_watermarked, prompts, messages)):
            image_no_watermark.save(os.path.join(cache_dir_no_watermark, f'data_{image_index}.png'))
            image_watermark.save(os.path.join(cache_dir_watermark, f'data_{image_index}.png'))
            image_inverse_watermark.save(os.path.join(cache_dir_inverse_watermark, f'data_{image_index}.png'))
            data_acc.append({'index': image_index, 'prompt': prompt, 'message': message.tolist()})
            image_index += 1

    # Save CSV
    df = pd.DataFrame(data_acc)
    df.to_csv(os.path.join(cache_dir, f'attack_dataset_{watermark_algorthim}', 'messages.csv'))
    print('Attack Dataset Created')

if __name__ == "__main__":
    torch.manual_seed(69)
    torch.cuda.manual_seed(69)
    torch.cuda.manual_seed_all(69)
    create_attack_dataset(1024, 16, 'cache', 'dwtdct')