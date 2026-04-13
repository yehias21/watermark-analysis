import os

import pandas as pd
import torch
import tqdm

from prompts_generator import next_prompt
from watermarks.PostProccessingWatermarksStableDiffusion import PostProccessingWatermarksStableDiffusion
from watermarks.TrwStableDiffusion import TrwStableDiffusion

# Module-level constants
RANDOM_SEED = 69
DEFAULT_NUM_SAMPLES = 1024
DEFAULT_BATCH_SIZE = 16
DEFAULT_CACHE_DIR = 'cache'
DEFAULT_WATERMARK = 'dwtdct'
SD_MODEL = "sd-legacy/stable-diffusion-v1-5"
RIVAGAN_DWT_MSG_LEN = 32
STEGASTAMP_MSG_LEN = 100
BIT_ALGORITHMS = ('rivagan', 'dwtdct', 'dwtdctsvd')


def _build_generator(watermark_algorthim: str):
    if watermark_algorthim == 'trw':
        return TrwStableDiffusion()
    return PostProccessingWatermarksStableDiffusion(model=SD_MODEL, watermark_algorthim=watermark_algorthim)


def _sample_messages(generator, watermark_algorthim: str, num_prompts: int):
    if watermark_algorthim == 'trw':
        generator.trw.set_message(generator.trw.sample_message(1)[0])
        return torch.repeat_interleave(
            generator.trw.get_message().to(generator.device).unsqueeze(0),
            num_prompts,
            dim=0,
        )
    if watermark_algorthim in BIT_ALGORITHMS:
        return torch.randint(0, 2, (num_prompts, RIVAGAN_DWT_MSG_LEN)).float()
    if watermark_algorthim == 'stegastamp':
        return torch.randint(0, 2, (num_prompts, STEGASTAMP_MSG_LEN)).float()
    return None


@torch.no_grad()
def create_attack_dataset(number_of_samples: int, batch_size: int, cache_dir: str, watermark_algorthim: str = 'rivagan'):
    prompt_iterator = next_prompt()
    base_dir = os.path.join(cache_dir, f'attack_dataset_{watermark_algorthim}')
    cache_dir_no_watermark = os.path.join(base_dir, 'no_watermark')
    cache_dir_watermark = os.path.join(base_dir, 'watermark')
    cache_dir_inverse_watermark = os.path.join(base_dir, 'inverse_watermark')
    os.makedirs(cache_dir_no_watermark, exist_ok=True)
    os.makedirs(cache_dir_watermark, exist_ok=True)
    os.makedirs(cache_dir_inverse_watermark, exist_ok=True)

    generator = _build_generator(watermark_algorthim)
    tqdm_bar = tqdm.tqdm(range(number_of_samples // batch_size), desc="Generating Attack Dataset")
    data_acc = []
    image_index = 0

    for _ in tqdm_bar:
        prompts = [next(prompt_iterator) for _ in range(batch_size)]
        messages = _sample_messages(generator, watermark_algorthim, len(prompts))
        images_no_watermarked, images_watermarked, images_inverse_watermarked = generator.generate_triplet(prompts, messages)

        for image_no_watermark, image_watermark, image_inverse_watermark, prompt, message in zip(
                images_no_watermarked, images_watermarked, images_inverse_watermarked, prompts, messages):
            image_no_watermark.save(os.path.join(cache_dir_no_watermark, f'data_{image_index}.png'))
            image_watermark.save(os.path.join(cache_dir_watermark, f'data_{image_index}.png'))
            image_inverse_watermark.save(os.path.join(cache_dir_inverse_watermark, f'data_{image_index}.png'))
            data_acc.append({'index': image_index, 'prompt': prompt, 'message': message.tolist()})
            image_index += 1

    # Save CSV
    df = pd.DataFrame(data_acc)
    df.to_csv(os.path.join(base_dir, 'messages.csv'))
    print('Attack Dataset Created')


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    create_attack_dataset(DEFAULT_NUM_SAMPLES, DEFAULT_BATCH_SIZE, DEFAULT_CACHE_DIR, DEFAULT_WATERMARK)
