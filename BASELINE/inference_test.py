import torchvision
import torch
from torch import nn

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

# from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import torch.optim as optim

import cv2
import random
import pandas as pd
import numpy as np
import math
import time
from tqdm import tqdm
import timm
import os

from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.metrics import classification_report

CFG = {
    'fold_num': 10,
    'seed': 719,
    'model_arch': 'vit_base_patch16_384',
    'img_size': 384,
    'epochs': 10,
    'train_bs': 16,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 2,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0'
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class MaskDataset(Dataset):
    def __init__(
        self, df, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
     #   self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = torch.tensor(self.df.loc[index,'class_label'])

        path = self.df.loc[index,'path']
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
       # print(type(img))
        if self.transforms:
            img = self.transforms(image=img)['image']
 #           img = img['image']
            
        if self.output_label == True:
            return img, target
        else:
            return img

def get_inference_transforms():
    return Compose([
            #CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
        A.Resize(height=CFG['img_size'], width = CFG['img_size']),
        A.Normalize(),
        ToTensorV2()
        ])

class MaskClassifier(nn.Module):
    def __init__(self, model_arch,n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, num_classes=n_class, pretrained=pretrained)
       # n_features = self.model.classifier.in_features
       # self.model.classifier = nn.Linear(n_features, n_class)
        
        # 초기화
        # torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        # stdv = 1. / math.sqrt(self.model.classifier.weight.size(1))
        # self.model.classifier.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.model(x)
        return x

def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [(image_preds.detach().cpu())]#[torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all



if __name__ == "__main__":
    seed_everything(CFG['seed'])
    train = pd.read_csv("/opt/ml/input/data/train/train_with_label.csv")
  
    f_name = '/opt/ml/input/data/eval/images' # inference image folder
    test_df = pd.read_csv("/opt/ml/input/data/eval/info.csv")
    test_df['path'] = test_df['ImageID'].apply(lambda x: "/".join([f_name,x]))

    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.class_label.values)
    
    for fold, (trn_idx, val_idx) in enumerate(folds):
    # we'll train fold 0 first
        if fold > 0:
            break 

        print('Inference fold {} started'.format(fold))

        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        valid_ds = MaskDataset(valid_, transforms=get_inference_transforms(), output_label=False)
        
        # test = pd.DataFrame()
        # test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
        test_ds = MaskDataset(test_df, transforms=get_inference_transforms(), output_label=False)
        
        val_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )
        
        tst_loader = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(CFG['device'])
        model = MaskClassifier(CFG['model_arch'],train['class_label'].nunique(), True)#.to(device)
        model = model.to(device)
        
        val_preds = []
        tst_preds = []


        model_folder = '/opt/ml/input/model_saver/vit_model'  # 모델 저장 폴더
        # 사용할 모델 리스트
        models = ['vit_base_patch16_384_fold_0_3_0.933',
        'vit_base_patch16_384_fold_0_4_0.971', 'vit_base_patch16_384_fold_0_6_0.988',
        'vit_base_patch16_384_fold_0_7_0.99', 'vit_base_patch16_384_fold_0_9_0.991']

        for i, model_version in enumerate(models):    
            model.load_state_dict(torch.load(model_folder+"/"+model_version))
            model = model.to(device)
            with torch.no_grad():
                val_preds += [inference_one_epoch(model, val_loader, device)]
                tst_preds += [inference_one_epoch(model, tst_loader, device)]
        
        val_preds = np.mean(val_preds, axis=0) 
        tst_preds = np.mean(tst_preds, axis=0) 
        
        print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.class_label.values, val_preds)))
        print('fold {} validation f1-score = {:.5f}'.format(fold, f1_score(valid_.class_label.values,np.argmax(val_preds, axis=1),average='macro')))
        print(classification_report(valid_.class_label, np.argmax(val_preds, axis=1)))

        submission = pd.read_csv("/opt/ml/input/data/eval/info.csv")
        submission['ans'] = np.argmax(tst_preds, axis=1)
        submission.to_csv("/opt/ml/input/code/submission_files/test_submission_vit.csv", index=False)

        del model, tst_loader, val_loader#, scaler, scheduler
        torch.cuda.empty_cache()