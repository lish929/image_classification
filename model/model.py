# -*- coding: utf-8 -*-
# @Time    : 2024/8/22 15:44
# @Author  : Shining
# @File    : model.py
# @Description :


"""
使用预训练模型，参考torchvision.models与timm.create_model()
创建的模型在最后两层或最后几层并没有形成统一
故在模型创建完成后 在原有fc 或 classifier层后添加特征层与分类层
"""

import timm
import torch
from torch import nn
from torchvision import models


class ClassificationModel(nn.Module):
    def __init__(self, model_name, feature_dim, class_num):
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.model = self._make_model()
        self._init_weights()

    def forward(self):
        pass

    def _make_model(self):
        if self.model_name in timm.list_models(pretrained=True):
            model = timm.create_model(self.model_name, pretrained=True)
        elif self.model_name in models.list_models():
            weight = models.get_model_weights(self.model_name)
            model = models.get_model(self.model_name)
            weights = weight.verify(weight.IMAGENET1K_V1)
            if weights is not None:
                model.load_state_dict(weights.get_state_dict(progress=True))
        feature_layer = nn.Sequential(
            nn.Linear(1000, self.feature_dim),
            nn.ReLU(inplace=True)
        )
        class_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.class_num)
        )
        model.add_module("feature_layer", feature_layer)
        model.add_module("class_layer", class_layer)

        return model

    def _init_weights(self):
        for item in self.model.named_children():
            if item[0] == "feature_layer":
                for m in item[1].modules():
                    if isinstance(m,nn.Linear):
                        nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain(("relu")))
            elif item[0] == "class_layer":
                if item[0] == "feature_layer":
                    for m in item[1].modules():
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_uniform(m.weight)


if __name__ == '__main__':
    # bat_resnext26ts.ch_in1k
    model = ClassificationModel("alexnet", 512, 7)
    # print(model)
