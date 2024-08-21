# -*- coding: utf-8 -*-
# @Time    : 2024/8/13 11:27
# @Author  : Shining
# @File    : dataset.py
# @Description :

import cv2
from multiprocessing.pool import ThreadPool
import numpy as np
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from transform import ClassificationTransformer

"""
root
    - category_1
        - file_1
        - file_2
        - ...
    - category_2
    - ...
    - ...
"""

ALLOW_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def check_image(file_path):
    try:
        image = Image.open(file_path)
        image.load()
        return True
    except Exception:
        return False


def iter_valid_files(directory, formats):
    walk = os.walk(directory)
    for root, _, files in sorted(walk, key=lambda x: x[0]):
        for file_name in sorted(files):
            file_path = os.path.join(directory, file_name)
            if file_name.lower().endswith(formats) and check_image(file_path):
                yield root, file_name


def index_subdirectory(directory, class_indices, formats):
    dir_name = os.path.basename(directory)
    valid_files = iter_valid_files(directory, formats)
    labels = []
    file_names = []
    for root, file_name in valid_files:
        # 根据类别名获取标签
        labels.append(class_indices[dir_name])
        absolute_path = os.path.join(dir_name, file_name)
        file_names.append(absolute_path)
    return file_names, labels


class ClassificationDataset(Dataset):
    def __init__(self, data_root, transformer, is_training=True):
        self.is_training = is_training
        self.transformer = transformer

        label_names = sorted(
            label_name for label_name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, label_name)))
        label_to_index = dict((label_name, index) for index, label_name in enumerate(label_names))

        pool = ThreadPool()
        results = []
        file_names = []
        labels = []

        for dir_path in (os.path.join(data_root, sub_dir) for sub_dir in label_names):
            results.append(
                pool.apply_async(
                    index_subdirectory,
                    (dir_path, label_to_index, ALLOW_FORMATS)
                )
            )

        for result in results:
            file_name, label = result.get()
            file_names = file_names + file_name
            labels = labels + label
        pool.close()
        image_paths = [os.path.join(data_root, file_name) for file_name in file_names]
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path, label = self.image_paths[index], self.labels[index]
        try:
            image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        except:
            raise RuntimeError("can not read image!")
        image = torch.from_numpy(image)
        label = torch.tensor(label)
        image, label = self.transformer(image, label, self.is_training)
        return image, label


if __name__ == '__main__':
    transformer = ClassificationTransformer([160, 160], 2, 0.0, 255.0, True, True)
    dataset = ClassificationDataset(r"C:\haoli\part-all\mask", transformer)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
