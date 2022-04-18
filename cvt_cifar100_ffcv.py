from torchvision import datasets
import os

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

def main(is_train, data_dir, write_path, max_resolution, num_workers, compress_probability):
    dataset = datasets.CIFAR100(root=data_dir, train=is_train, download=False)

    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode='raw',
                               max_resolution=max_resolution,
                               compress_probability=compress_probability),
        'label': IntField(),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(dataset)

if __name__ == '__main__':
    for split in ['train', 'val']:
        data_dir = '/srv/share/gstoica3/datasets/cifar-100-python'
        max_resolution = 224
        num_workers = 16
        compress_probability = 0

        is_train = split == 'train'
        ffcv_split = 'train' if is_train else 'val'
        write_path = os.path.join('/srv/share4/thearn6/datasets/ffcv_cifar100', f'{ffcv_split}_{max_resolution}_{compress_probability}_raw.ffcv')
        main(is_train, data_dir, write_path, max_resolution, num_workers, compress_probability)

