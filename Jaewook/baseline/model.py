import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchvision import models
import timm
import math

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

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyResnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        initialize_weights(self.model.fc)
        

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

class EffB4Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
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

class EffB3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1536, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )
        initialize_weights(self.model.classifier)
        
    def forward(self, x):
        x = self.model(x)
        return x

class VitBase(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True)
        self.model.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )
        initialize_weights(self.model.head)
        
    def forward(self, x):
        x = self.model(x)
        return x

class NGModel(nn.Module):
    def __init__(self, num_classes):
        super(NGModel, self).__init__()
        model = models.vgg19(pretrained=True)
        self.features = model.features
        self.dropout=nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512,num_classes)
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

class JHModel(nn.Module):  # transformer model
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            'vit_base_r50_s16_384', pretrained=pretrained)
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=num_classes),
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


if __name__ == '__main__':
    print('model.py is running')