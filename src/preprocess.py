import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

def get_dataloader(config):
    dataset_config = config['data']
    dataset_name = dataset_config['name']
    batch_size = config['training']['batch_size']
    image_size = dataset_config['image_size']

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print('Warning: HF_TOKEN environment variable not set. Downloads may fail for gated datasets.')

    try:
        if dataset_name == 'cifar10':
            dataset = load_dataset('uoft-cs/cifar10', split='train', token=hf_token)
        elif dataset_name == 'celeba-hq':
            dataset = load_dataset('mattymchen/celeba-hq', split='train', token=hf_token)
        elif dataset_name == 'imagenet-64':
             dataset = load_dataset('gitpull/Imagenet_64x64', split='train', token=hf_token)
        elif dataset_name == 'lsun-bedroom':
            dataset = load_dataset('fformosa/LSUN_bedroom_whole', split='train', token=hf_token)
        else:
            raise ValueError(f'Unknown dataset: {dataset_name}')
    except Exception as e:
        print(f'Failed to load dataset {dataset_name}. Please check dataset name and network connection.')
        raise FileNotFoundError(f'Could not download or process {dataset_name}.') from e

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    def transform(examples):
        images = [preprocess(image.convert('RGB')) for image in examples['img']]
        return {'img': torch.stack(images)}

    dataset.set_transform(transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
