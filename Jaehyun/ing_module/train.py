from sre_constants import OP_IGNORE
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# from torchvision import transforms
from torch.cuda.amp import GradScaler

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

import pandas as pd
import numpy as np
import os
import random

from sklearn.model_selection import StratifiedKFold
from make_df_sep_val_trn import SepValidTrain
from torch.utils.tensorboard import SummaryWriter
from data_utils.datasets import MaskDataset
from data_utils.data_loaders import MaskDataLoader
from model.models import MaskClassifier_efficient, MaskClassifier_transformer
from trainer import Trainer

CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_class': 'Transformer',
    # 'model_class': 'EfficientNet',
    'model_arch': 'swin_base_patch4_window12_384',
    'img_size': 384,
    'epochs': 10,
    'train_bs': 32,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 2,
    # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'accum_iter': 2,
    'verbose_step': 1,
    'device': 'cuda:0',
    'saved_floder': 'test',
    'config_BETA': 0.5,
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_transforms():
    return Compose([
        A.Resize(height=CFG['img_size'], width=CFG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.RandomFog(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RGBShift(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.GaussNoise(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])


def get_valid_transforms():
    return Compose([
        A.Resize(height=CFG['img_size'], width=CFG['img_size']),
        A.Normalize(),
        ToTensorV2()
    ])


def make_save_dir(path, filename):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists('{}/{}'.format(path, filename)):
        os.mkdir('{}/{}'.format(path, filename))


if __name__ == "__main__":
    seed_everything(CFG['seed'])

    # Tensorboard
    logger = SummaryWriter(log_dir='logs/{}'.format(CFG['saved_floder']))

    make_save_dir(
        '/opt/ml/image-classification-level1-31/Jaehyun/saved_model/', CFG['saved_floder'])
    train = pd.read_csv('/opt/ml/input/data/train/train2.csv')
    valid = pd.read_csv('/opt/ml/input/data/train/valid.csv')

    raw_train = SepValidTrain().make_tmp_labeled_df()

    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(
        np.arange(raw_train.shape[0]), raw_train.tmp_label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break

        train_ = raw_train.loc[trn_idx, :].reset_index(drop=True)
        valid_ = raw_train.loc[val_idx, :].reset_index(drop=True)

        # 데이터 셋 & 데이터 로더 선언
        train = SepValidTrain().make_detailpath_N_label_df(train_)
        valid = SepValidTrain().make_detailpath_N_label_df(valid_)

        print('Training with {} started'.format(fold))
        train_set = MaskDataset(train, get_train_transforms())
        valid_set = MaskDataset(valid, get_train_transforms())

        train_loader = MaskDataLoader(
            train_set, CFG['train_bs'], num_workers=CFG['num_workers'], sampler='WeightedRandomSampler')
        val_loader = MaskDataLoader(
            valid_set, CFG['valid_bs'], num_workers=CFG['num_workers'])

        device = torch.device(CFG['device'])

        # 모델 생성
        if CFG['model_class'] == 'Transformer':
            print('Training Model is {}'.format(CFG['model_class']))
            model = MaskClassifier_transformer(
                CFG['model_arch'], train['class_label'].nunique(), True)
        elif CFG['model_class'] == 'EfficientNet':
            print('Training Model is {}'.format(CFG['model_class']))
            model = MaskClassifier_efficient(
                CFG['model_arch'], train['class_label'].nunique(), True)
        else:
            raise Exception('Check the model name again')
        model = model.to(device)

        # Auto Cast를 위한 Scaler, optimizer, scheduler 선언
        scaler = GradScaler()
        optimizer = optim.Adam(
            model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)

        # loss 선언
        loss_fn = nn.CrossEntropyLoss().to(device)

        best_valid_f1 = 0.7

        # Trainer 선언
        trainer = Trainer(model=model,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          device=device,
                          scaler=scaler,
                          logger=logger,
                          scheduler=scheduler
                          )
        # 학습 시작
        for epoch in range(CFG['epochs']):
            trainer.train_one_epoch(epoch, train_loader, cutmix_beta=0.5)
            with torch.no_grad():
                valid_f1 = trainer.valid_one_epoch(epoch, val_loader)
            folder_path = os.path.join(
                '/opt/ml/image-classification-level1-31/Jaehyun/saved_model/', CFG['saved_floder'])
            if best_valid_f1 < valid_f1:
                torch.save(model.state_dict(), os.path.join(folder_path, '{}_fold_{}_{}_{}.pt'.format(
                    CFG['model_arch'], fold, epoch, np.round(valid_f1, 3))))
                best_valid_f1 = valid_f1
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()