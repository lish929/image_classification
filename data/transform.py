# -*- coding: utf-8 -*-
# @Time    : 2024/8/21/21 9:27
# @Author  : Shining
# @File    : transform.py
# @Description :


import torch
from torch.nn.functional import one_hot
from torchvision.transforms import InterpolationMode,functional as F


class ClassificationTransformer(torch.nn.Module):
    def __init__(self,input_shape,num_classes,mean,std,usr_augmentation=False,use_padding=True):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.mean = mean
        self.std = std
        self.use_augmentation = usr_augmentation
        self.use_padding = use_padding

    def __call__(self, image,label,is_training):
        if self.use_augmentation:
            return self._transform_with_augmentation(image=image,label=label,is_training=is_training)
        else:
            return self._transform_without_augmentation(image,label)
    def _transform_with_augmentation(self,image,label,is_training):
        image = torch.permute(image, (2, 1, 0))
        if is_training:
            image = self._transform_for_train(image,use_padding=self.use_padding)
        else:
            image = self._transform_for_eval(image,use_padding=self.use_padding)

        image = (image-self.mean)/self.std
        label = one_hot(label,self.num_classes)
        return image,label

    def _transform_without_augmentation(self,image,label):
        image = torch.permute(image, (2, 0, 1))
        image = Resize(self.input_shape)(image)
        image = (image - self.mean) / self.std
        label = one_hot(label, self.num_classes)
        return image, label

    def _transform_for_train(self,image,use_padding=False):
        if use_padding:
            image = Pad()(image)
        image = RandomCrop()(image)
        image = Flip()(image)
        image = Resize(size=self.input_shape)(image)
        return image

    def _transform_for_eval(self,image,use_padding=False):
        if use_padding:
            image = Pad()(image)
        else:
            image = CenterCrop()(image)
        image = Resize(self.input_shape)(image)
        return image


class Pad(torch.nn.Module):
    def __init__(self,fill=255, padding_mode="constant"):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(image):

        _, h, w = F.get_dimensions(image)

        max_len = max(h,w)
        top = (max_len-h)//2
        bottom = max_len-h-top
        left = (max_len-w)//2
        right = max_len-w-left
        # left, top, right and bottom
        return [left,top,right,bottom]

    def forward(self, image):
        return F.pad(image, self.get_params(image), self.fill, self.padding_mode)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

class RandomCrop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_params(image, ratio=0.7):
        _, h, w = F.get_dimensions(image)
        th, tw = int(h*ratio),int(w*ratio)

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def forward(self, image):
        i, j, h, w = self.get_params(image)

        return F.crop(image, i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

class Flip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
        if torch.rand(1) < self.p:
            image = F.vflip(image)
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, image):
        return F.resize(image, self.size, self.interpolation)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class CenterCrop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_params(image,ratio=0.7):
        _, h, w = F.get_dimensions(image)
        th, tw = torch.randint(int(h * ratio),h,size=(1,)).item(), torch.randint(int(w * ratio),w,size=(1,)).item()
        return [th, tw]

    def forward(self, image):
        return F.center_crop(image, self.get_params(image))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"