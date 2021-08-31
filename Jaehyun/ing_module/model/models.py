import torch
import timm
import torch.nn as nn


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
