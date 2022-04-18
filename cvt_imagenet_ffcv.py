from torchvision import datasets
import os

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

def main(split, data_dir, write_path, max_resolution, num_workers, jpeg_quality, compress_probability):
    dataset = datasets.ImageFolder(data_dir)

    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode='smart',
                               max_resolution=max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        'label': IntField(),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(dataset)

if __name__ == '__main__':
    for split in ['train', 'val']:
        data_dir = os.path.join('/srv/datasets/ImageNet', split)
        max_resolution = 224
        num_workers = 16
        jpeg_quality = 90
        #compress_probability = 0.5

        for compress_probability in [0, 1]:
            write_path = os.path.join('/srv/share4/thearn6/datasets/ffcv_imagenet2', f'{split}_{max_resolution}_{compress_probability}_{jpeg_quality}.ffcv')
            main(split, data_dir, write_path, max_resolution, num_workers, jpeg_quality, compress_probability)

