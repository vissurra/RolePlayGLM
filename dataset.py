import json
import os
import random

from loguru import logger

from dataset_template import Template

with open('data/creator_names.txt', 'r') as creator_file:
    creator_names = [line.strip() for line in creator_file.readlines()]


def build(folder):
    data = []
    for i in range(200):
        template = Template.random_init()
        q, a = template.get_character_qa()
        data.append({
            'input': q,
            'output': a,
            'history': template.build_instruction_history(random.choice(creator_names))
        })

    save(folder, data)


def save(folder, data: list[dict]):
    os.makedirs(folder, exist_ok=True)

    train_size = int(len(data) * 0.85)
    train_data = data[:train_size]
    valid_data = data[train_size:]

    logger.info(f'train size: {len(train_data)}')
    with open(f'{folder}/train.json', 'w') as f:
        for item in train_data:
            item = json.dumps(item, ensure_ascii=False)
            f.write(item)
            f.write('\n')

    logger.info(f'valid size: {len(valid_data)}')
    with open(f'{folder}/valid.json', 'w') as f:
        for item in valid_data:
            item = json.dumps(item, ensure_ascii=False)
            f.write(item)
            f.write('\n')


if __name__ == '__main__':
    build('data/dataset')
