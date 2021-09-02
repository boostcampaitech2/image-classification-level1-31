import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from importlib import import_module
import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
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
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('inception_v3', pretrained = True)
        self.classifier = nn.Linear(1000,18)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        x = self.classifier(x)
        return x


class MultiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.agemodel = torch.load('model/age/best.pt') #age trained model
        self.gendermodel = torch.load('model/gender/best.pt') #gender trained model
        self.maskmodel = torch.load('model/mask/best.pt') #mask trained model
        self.classifier = nn.Linear(8,18)
    def forward(self, x):
        
        age = self.agemodel(x)
        age = torch.argmax(age, 1)
        gender = self.gendermodel(x)
        gender = torch.argmax(gender, 1)
        mask = self.maskmodel(x)
        mask = torch.argmax(mask, 1)

        y = age + gender*3 + mask*6
        y = y.cpu().numpy()
        # output = torch.zeros(18)
        # output[y] = 1
        return y

        # age = self.agemodel(x)
        # gender = self.gendermodel(x)
        # mask = self.maskmodel(x)
        # output = torch.cat([age,gender,mask])
        # output = self.classifier(output)

        # return output



class GenderModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('inception_v3', pretrained = True)
        self.classifier = nn.Linear(1000,2)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        x = self.classifier(x)
        return x

class AgeModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('inception_v3', pretrained = True)
        self.classifier = nn.Linear(1000,3)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        x = self.classifier(x)
        return x

class agemodel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.load('model/age/best.pt')
        
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x
class maskmodel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.load('model/mask/best.pt')
        
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x
class gendermodel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.load('model/gender/best.pt')
        
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x

class MaskModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('inception_v3', pretrained = True)
        self.classifier = nn.Linear(1000,3)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        x = self.classifier(x)
        return x
model = MultiModel
model()