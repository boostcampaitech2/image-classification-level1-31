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
    'model_arch': ['tf_efficientnet_b4_ns', 'seresnext50_32x4d', 'vit_base_patch16_224', 'eca_nfnet_l1', 'regnety_032'],
    'img_size': [380, 224, 224, 320, 288],
    'epochs': 10,
    'train_bs': 32,
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
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        path = self.df.loc[index,'path']
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # img = img[:,:,::-1]
        
        # xmin = int(self.df.loc[index,'min_x'])
        # ymin = int(self.df.loc[index,'min_y'])
        # xmax = int(self.df.loc[index,'max_x'])
        # ymax = int(self.df.loc[index,'max_y'])

        # if xmin < 0: xmin = 0
        # if ymin < 0: ymin = 0
        # if xmax > 384: xmax = 384
        # if ymax > 512: ymax = 512

        # img = img[ymin:ymax,xmin:xmax,:]
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.output_label == True:
            target = torch.tensor(self.df.loc[index,'class_label'])
            return img, target
        else:
            return img

def get_inference_transforms(img_size = 224):
    return Compose([
        A.CenterCrop(height=380,width =260, always_apply=True, p=1.0),
        A.Resize(height=img_size, width = img_size),
        A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), p=1),
        ToTensorV2()
        ])

class MaskClassifier(nn.Module):
    def __init__(self, model_arch,n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, num_classes=n_class, pretrained=pretrained)
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
    train = pd.read_csv("/opt/ml/input/data/train/train_label_box.csv")
  
    train['person_id'] = train['path'].apply(lambda x: x.split("/")[-2].split("_")[0])
    train['specific_age'] = train['path'].apply(lambda x: int(x.split("/")[-2].split("_")[-1]))
    temp_train = train[['Gender', "specific_age", "person_id"]].drop_duplicates().reset_index(drop=True)
    temp_train[['re_Age']] = temp_train['specific_age'].apply(lambda x: '<30' if x < 30 else (">=30 and <60" if x < 60 else ">=60"))


    f_name = '/opt/ml/input/data/eval/images' # inference image folder
    test_df = pd.read_csv("/opt/ml/input/data/eval/info.csv")
    test_df['path'] = test_df['ImageID'].apply(lambda x: "/".join([f_name,x]))

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(temp_train.shape[0]), temp_train['Gender'] + temp_train['re_Age'])    
    

    tst_preds = []
    validation_F1_list = []

    model_folder = '/opt/ml/input/model_saver'  # 모델 저장 폴더
    mode_name  = ['efficientnet_b4_ns', 'resnext50', 'vit_with_box', 'nfnet', 'regnet']
    models = ['tf_efficientnet_b4_ns_fold_0_45_0.838', 'seresnext50_32x4d_fold_1_25_0.781', 
    'vit_base_patch16_224_fold_2_38_0.802', 'eca_nfnet_l1_fold_3_46_0.848', 'regnety_032_fold_4_35_0.813']


    for fold, (trn_idx, val_idx) in enumerate(folds):
        val_preds = []
        print('Inference fold {} started'.format(fold))
        temp_train_split = temp_train.loc[trn_idx]
        train_list = list(temp_train_split['person_id'] + "_" + temp_train_split['Gender'].apply(lambda x: x.lower()) + "_" +  "Asian" + "_" + temp_train_split['specific_age'].astype(str))
        train['train_valid'] = train['path'].apply(lambda x:'train' if x.split("/")[-2] in train_list else 'valid')
        trn_idx = train.loc[train['train_valid'] == 'train'].index
        val_idx = train.loc[train['train_valid'] != 'train'].index

        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        valid_ds = MaskDataset(valid_, transforms=get_inference_transforms(img_size = CFG['img_size'][fold]), output_label=False)
    
        test_ds = MaskDataset(test_df, transforms=get_inference_transforms(img_size = CFG['img_size'][fold]), output_label=False)
        
        val_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False
        )
        
        tst_loader = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(CFG['device'])
        model = MaskClassifier(CFG['model_arch'][fold],train['class_label'].nunique(), True)#.to(device)
        model = model.to(device)
        model.load_state_dict(torch.load("/".join([model_folder, mode_name[fold],models[fold]])))

        with torch.no_grad():
            val_preds += [inference_one_epoch(model, val_loader, device)]
            tst_preds += [inference_one_epoch(model, tst_loader, device)]

        val_preds = np.mean(val_preds, axis=0)    
        print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.class_label.values, val_preds)))
        print('fold {} validation f1-score = {:.5f}'.format(fold, f1_score(valid_.class_label.values,np.argmax(val_preds, axis=1),average='macro')))
        validation_F1_list.append(f1_score(valid_.class_label.values,np.argmax(val_preds, axis=1),average='macro'))
        del model, tst_loader, val_loader#, scaler, scheduler
        torch.cuda.empty_cache()
    
    print(f"CV F1 score: {np.mean(validation_F1_list)}")
    tst_preds = np.mean(tst_preds, axis=0) 
    print("예측 후 평균: ",tst_preds.shape)
    print("예측 후 평균 argmax: ",np.argmax(tst_preds, axis=1).shape)

    submission = pd.read_csv("/opt/ml/input/data/eval/info.csv")
    submission['ans'] = np.argmax(tst_preds, axis=1)
    submission.to_csv("/opt/ml/input/code/submission_files/SEO_DJ_FINAL_submission.csv", index=False)
