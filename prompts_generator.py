import json
import random

# Module-level constants
RANDOM_SEED = 69
COCO_PATH = './coco.json'


def next_prompt():
    """Generator that yields prompts from the dataset."""
    random.seed(RANDOM_SEED)
    with open(COCO_PATH) as f:
        dataset = json.load(f)['annotations']
    random.shuffle(dataset)
    for row in dataset:
        yield row['caption']
