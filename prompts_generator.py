import json
import random


def next_prompt():
    """
    Generator that yields prompts from the dataset.
    """
    random.seed(101)
    with open('./coco.json') as f:
        dataset = json.load(f)['annotations']
        random.shuffle(dataset)
        for row in dataset:
            yield row['caption']