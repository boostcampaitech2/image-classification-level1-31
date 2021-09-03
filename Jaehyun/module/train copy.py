import torch
from torch import nn
import torch.optim as optim

from torch.cuda.amp import GradScaler

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

import pandas as pd
import numpy as np
import os
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from torch.utils.tensorboard import SummaryWriter
from data_utils.datasets import MaskDataset
from data_utils.data_loaders import MaskDataLoader
from data_utils.make_df_sep_val_trn import SepValidTrain
from model.models import *
from trainer import Trainer, CosineAnnealingWarmUpRestarts
from model.loss import *


model_class = {'swin_base_patch4_window12_384': 'Transformer',
               'tf_efficientnet_b4_ns':  'EfficientNet',
               'MaskClassifier_resnet50': 'ResNet50',
               'vit_base_r50_s16_384': 'Transformer',
               }

CFG = {
    'model_arch': 'vit_base_r50_s16_384',
    'saved_floder': 'vit_base_r50_s16_384_last',
    'loss': 'crossentropy',
    # 'loss': 'f1',
    # 'loss': 'labelsmooth',
    'img_size': 384,
    'train_bs': 16,
    'valid_bs': 32,
    'fold_num': 5,
    'seed': 719,
    # 'warmup_epochs': 10,
    'epochs': 10,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'accum_iter': 2,
    'verbose_step': 1,
    'device': 'cuda:0',
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
        # A.ShiftScaleRotate(p=0.5),
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
    make_save_dir(
        '/opt/ml/image-classification-level1-31/Jaehyun/saved_model/', CFG['saved_floder'])

    # config txt로 저장
    f = open(os.path.join('/opt/ml/image-classification-level1-31/Jaehyun/saved_model/' +
             CFG['saved_floder'], 'log.txt'), 'w')
    f.write(str(CFG.items()).replace(',', '\n'))
    f.close()

    train = pd.read_csv('/opt/ml/input/data/train/train3.csv')

    raw_train = SepValidTrain().make_tmp_labeled_df()

    # if fold > 0:
    #     break

    # Tensorboard
    logger = SummaryWriter(
        log_dir='logs/{}/{}'.format(CFG['saved_floder'], 0))

    train_ = raw_train.loc[:, :].reset_index(drop=True)

    # 데이터 셋 & 데이터 로더 선언
    train = SepValidTrain().make_detailpath_N_label_df(train_)

    # 추가 데이터 60-98 데이터 학습
    new_train_df = pd.read_csv('/opt/ml/input/data/train/newim.csv')
    new_train_df = new_train_df.drop(['Unnamed: 0'], axis=1)
    train = pd.concat([train, new_train_df], axis=0)

    print('Training with {} started'.format(0))
    train_set = MaskDataset(train, get_train_transforms())

    train_loader = MaskDataLoader(
        train_set, CFG['train_bs'],
        num_workers=CFG['num_workers'],
        sampler='WeightedRandomSampler',
        # sampler='BalanceClassSampler'
    )

    device = torch.device(CFG['device'])

    # 모델 생성

    # if model_class[CFG['model_arch']] == 'Transformer':
    #     print('Training Model is {}'.format(
    #         model_class[CFG['model_arch']]))
    #     model = MaskClassifier_transformer(
    #         CFG['model_arch'], train['class_label'].nunique(), True)

    if model_class[CFG['model_arch']] == 'Transformer':
        print('Training Model is {}'.format(
            model_class[CFG['model_arch']]))
        model = MaskClassifier_custom_transformer(
            CFG['model_arch'], train['class_label'].nunique(), True)

    elif model_class[CFG['model_arch']] == 'EfficientNet':
        print('Training Model is {}'.format(
            model_class[CFG['model_arch']]))
        model = MaskClassifier_custom_efficient(
            CFG['model_arch'], train['class_label'].nunique(), True)

    elif model_class[CFG['model_arch']] == 'ResNet50':
        print('Training Model is {}'.format(
            model_class[CFG['model_arch']]))
        model = MaskClassifier_resnet50(
            CFG['model_arch'], train['class_label'].nunique(), True)

    else:
        raise Exception('Check the model name again')
    model = model.to(device)

    # Auto Cast를 위한 Scaler, optimizer, scheduler 선언
    scaler = GradScaler()
    optimizer = optim.Adam(
        model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
    # scheduler = CosineAnnealingWarmUpRestarts(
    #     optimizer, T_0=CFG['T_0'], T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5, last_epoch=-1)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5)

    # loss 선언
    if CFG['loss'] == 'crossentropy':
        # class_weights = class_weight.compute_class_weight(
        #     class_weight='balanced', classes=np.arange(18), y=train['class_label'].values)
        # loss_fn = nn.CrossEntropyLoss(
        #     weight=torch.tensor(class_weights, dtype=torch.float)
        # ).to(device)
        loss_fn = nn.CrossEntropyLoss()
    elif CFG['loss'] == 'f1':
        loss_fn = F1Loss(18)
    elif CFG['loss'] == 'labelsmooth':
        loss_fn = LabelSmoothingLoss(18, 0.2)

    loss_fn.to(device)
    best_valid_f1 = 0.7

    # Trainer 선언
    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      device=device,
                      scaler=scaler,
                      logger=logger,
                      scheduler=scheduler,
                      schd_batch_update=False
                      )
    # 학습 시작
    # for epoch in range(CFG['warmup_epochs']):
    #     trainer.train_one_epoch(
    #         epoch, train_loader, cutmix_beta=0.5, accum_iter=2)
    #     with torch.no_grad():
    #         valid_f1 = trainer.valid_one_epoch(epoch, val_loader)
    #     folder_path = os.path.join(
    #         '/opt/ml/image-classification-level1-31/Jaehyun/saved_model/', CFG['saved_floder'])
    #     if best_valid_f1 < valid_f1:
    #         torch.save(model.state_dict(), os.path.join(folder_path, '{}_fold_{}_{}_{}.pt'.format(
    #             CFG['model_arch'], fold, epoch, np.round(valid_f1, 3))))
    #         best_valid_f1 = valid_f1

    # trainer.loss_fn = F1Loss(18)
    for epoch in range(CFG['epochs']):
        trainer.train_one_epoch(
            epoch, train_loader, cutmix_beta=0.5, accum_iter=2)
        folder_path = os.path.join(
            '/opt/ml/image-classification-level1-31/Jaehyun/saved_model/', CFG['saved_floder'])
        torch.save(model.state_dict(), os.path.join(folder_path, '{}_fold_{}_{}_{}.pt'.format(
            CFG['model_arch'], 0, epoch, np.round(0, 3))))
    del model, optimizer, train_loader, scaler, scheduler
    torch.cuda.empty_cache()
