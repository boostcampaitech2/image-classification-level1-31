from symbol import parameters
import torch
import timm
import torch.nn as nn
import math
from torchvision import models
import torch.nn.init as init


class MaskClassifier(nn.Module):  # transformer model
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch, num_classes=n_class, pretrained=pretrained)
        # in_feature = self.model.head.in_features
        # self.model.head = nn.Linear(
        #     in_features=in_feature, out_features=n_class)
        # 초기화 모델에 따라 마지막단이 (이름이) classifier가 아닐 수 있습니다.
        # torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        # stdv = 1. / math.sqrt(self.model.classifier.weight.size(1))
        # self.model.classifier.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.model(x)
        return x


class MaskClassifier_efficient(nn.Module):  # efficientnet model
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch, num_classes=n_class, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, n_class)

        # 초기화 모델에 따라 마지막단이 (이름이) classifier가 아닐 수 있습니다.
        # torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        # stdv = 1. / math.sqrt(self.model.classifier.weight.size(1))
        # self.model.classifier.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.model(x)
        return x


class MaskClassifier_transformer(nn.Module):  # transformer model
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained)
        in_feature = self.model.head.in_features
        self.model.head = nn.Linear(
            in_features=in_feature, out_features=n_class)
        # 초기화 모델에 따라 마지막단이 (이름이) classifier가 아닐 수 있습니다.
        # torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        # stdv = 1. / math.sqrt(self.model.classifier.weight.size(1))
        # self.model.classifier.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.model(x)
        return x


class MaskClassifier_custom_efficient(nn.Module):  # efficientnet model
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch, num_classes=n_class, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(in_features, n_class)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=n_class),
        )
        # 초기화 모델에 따라 마지막단이 (이름이) classifier가 아닐 수 있습니다.

        def my_xavier_uniform(submodule):
            if isinstance(submodule, nn.Linear):
                torch.nn.init.xavier_uniform_(submodule.weight)
                stdv = 1. / math.sqrt(submodule.weight.size(1))
                submodule.bias.data.uniform_(-stdv, stdv)
        self.model.classifier.apply(my_xavier_uniform)

    def forward(self, x):
        x = self.model(x)
        return x


class MaskClassifier_custom_transformer(nn.Module):  # transformer model
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained)
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=n_class),
        )

        def my_xavier_uniform(submodule):
            if isinstance(submodule, nn.Linear):
                torch.nn.init.xavier_uniform_(submodule.weight)
                stdv = 1. / math.sqrt(submodule.weight.size(1))
                submodule.bias.data.uniform_(-stdv, stdv)
        self.model.head.apply(my_xavier_uniform)

    def forward(self, x):
        x = self.model(x)
        return x


class MyModel(nn.Module):  # 나경님 vgg19 모델
    def __init__(self, num_classes: int = 1000):
        super(MyModel, self).__init__()
        model = models.vgg19(pretrained=True)
        self.features = model.features
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TestModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model(
            'tf_efficientnet_b4_ns', pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
        initialize_weights(self.model.classifier)

    def forward(self, x):
        x = self.model(x)
        return x


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
