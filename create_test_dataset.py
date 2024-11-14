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
    cache_dir_watermark = os.path.join(cache_dir, f'test_dataset_{watermark_algorthim}')
    os.makedirs(cache_dir_watermark, exist_ok=True)
    if watermark_algorthim == 'trw':
        wm = TrwStableDiffusion()
    else:
        wm = PostProccessingWatermarksStableDiffusion(watermark_algorthim=watermark_algorthim)
    tqdm_bar = tqdm.tqdm(range(number_of_samples // batch_size), desc="Generating Test Dataset")
    data_acc = []
    image_index = 0

    for batch in tqdm_bar:
        print(f'Batch {batch + 1}/{number_of_samples // batch_size}')
        prompts = [next(prompt_iterator) for _ in range(batch_size)]
        messages = torch.randint(0, 2, (len(prompts), 32)).float()
        images = wm.generate(prompts, watermark=True, messages=messages)
        
        for i, (image, prompt, message) in enumerate(
                zip(images, prompts, messages)):
            image.save(os.path.join(cache_dir_watermark, f'data_{image_index}.png'))
            data_acc.append({'index': image_index, 'prompt': prompt, 'message': message.tolist()})
            image_index += 1

    # Save CSV
    df = pd.DataFrame(data_acc)
    df.to_csv(os.path.join(cache_dir, f'test_dataset_{watermark_algorthim}', 'messages.csv'))
    print('Attack Dataset Created')

if __name__ == "__main__":
    create_attack_dataset(1024, 16, 'cache', 'trw')