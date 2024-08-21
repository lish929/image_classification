# -*- coding: utf-8 -*-
# @Time    : 2024/8/13/ 9:53
# @Author  : Shining
# @File    : train.py.py
# @Description :
import logging

from timm.data.dataset_factory import create_dataset
import torch

from utils import *


def get_dataset(config):

    if string_is_space_or_empty(config.dataset.train_dataset_dir):
        raise RuntimeError("training dataset is required to train the model!")



if __name__ == '__main__':
    get_dataset("")