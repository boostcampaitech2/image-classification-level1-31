import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss


import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
import re
import os
import math
import time
import random

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from make_df_sep_val_trn import SepValidTrain
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report


CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'cait_s36_384',
    'img_size': 384,
    'epochs': 10,
    'train_bs': 16,
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
    'saved_file_name': 'cait_s36_384_kflod',
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


def make_df_with_label(data_folder_directory="/opt/ml/input/data/train"):
    """
    너무 복잡하다.....
    data_folder_directory 주소를 학습이미지가 들어있는 폴더로 설정!
    class_mapping.csv가 train.csv와 동일한 위치에 있어야합니다.
    """

    train = pd.read_csv(data_folder_directory + "/" + "train.csv")  # 원본파일
    try:
        train_label = pd.read_csv(
            "/".join([data_folder_directory, "class_mapping.csv"]))
    except:
        print(f"{data_folder_directory}에 class mapping csv가 없습니다.\n")

    total_img_list = []  # 이미지 저장할 주소
    current_image_folder_directory = data_folder_directory + '/images'  # 학습이미지가 들어있는 폴더
    for folder_name in train['path']:  # train['path']에는 한사람에 대한 이미지 폴더가 들어있다.
        temp_dir = "/".join([current_image_folder_directory, folder_name])
        # 해당 폴더에 있는 "._" 으로 시작하지 않는 모든 파일 불러오기 -> 7장의 사진을 불러온다.
        total_img_list.extend(["/".join([temp_dir, file_name])
                              for file_name in os.listdir(temp_dir) if not file_name.startswith('._')])

    train_df = pd.DataFrame({'path': total_img_list})
    train_df['Gender'] = train_df['path'].apply(
        lambda x: x.split("/")[-2].split("_")[1].capitalize())
    train_df['Age'] = train_df['path'].apply(
        lambda x: int(x.split("/")[-2].split("_")[-1]))
    train_df['Mask'] = train_df['path'].apply(
        lambda x: re.sub(r'[0-9]', '', x.split("/")[-1].split(".")[0]))

    # 나이 재변환
    train_df['Age'] = train_df['Age'].apply(
        lambda x: '<30' if x < 30 else (">=30 and <60" if x < 60 else ">=60"))
    train_df['Mask'] = train_df['Mask'].map(
        {'incorrect_mask': 'Incorrect', 'mask': "Wear", "normal": "Not Wear"})

    train_df = train_df.merge(train_label, how='left', on=[
                              'Gender', 'Age', 'Mask'])

    return train_df


class MaskDataset(Dataset):
    def __init__(
        self, df, transforms=None, output_label=True
    ):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.output_label = output_label
        self.labels = self.df['class_label']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        path = self.df.loc[index, 'path']
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # img = img[:,:,::-1]

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label == True:
            target = torch.tensor(self.df.loc[index, 'class_label'])
            return img, target
        else:
            return img


def get_train_transforms():
    return Compose([
        A.Resize(height=CFG['img_size'], width=CFG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.RandomFog(p=0.5),
        # A.ShiftScaleRotate(p=0.5),
        # A.RGBShift(p=0.5),
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


def prepare_dataloader(df_train, df_valid):
    # ? from catalyst.data.sampler import BalanceClassSampler
    # targets = df.class_label
    # class_count = np.unique(targets, return_counts=True)[1]

    # weight = 1. / class_count
    # samples_weight = weight[targets]
    # samples_weight = torch.from_numpy(samples_weight)
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    #

    train_ = df_train
    valid_ = df_valid

    train_ds = MaskDataset(
        train_, transforms=get_train_transforms(), output_label=True)
    valid_ds = MaskDataset(
        valid_, transforms=get_valid_transforms(), output_label=True)

    targets = train_ds.labels
    class_count = np.unique(targets, return_counts=True)[1]

    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler_train = WeightedRandomSampler(samples_weight, len(samples_weight))

    targets = valid_ds.labels
    class_count = np.unique(targets, return_counts=True)[1]

    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler_valid = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        # shuffle=True,
        num_workers=CFG['num_workers'],
        sampler=sampler_train
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    return train_loader, val_loader


class MaskClassifier(nn.Module):  # efficientnet model
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch, num_classes=n_class, pretrained=pretrained)
       # n_features = self.model.classifier.in_features
       # self.model.classifier = nn.Linear(n_features, n_class)

        # 초기화 모델에 따라 마지막단이 (이름이) classifier가 아닐 수 있습니다.
        # torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        # stdv = 1. / math.sqrt(self.model.classifier.weight.size(1))
        # self.model.classifier.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.model(x)
        return x


# class MaskClassifier(nn.Module):  # transfer model
#     def __init__(self, model_arch, n_class, pretrained=False):
#         super().__init__()
#         self.model = timm.create_model(
#             model_arch, pretrained=pretrained)
#         in_feature = self.model.head.in_features
#         self.model.head = nn.Linear(in_features=in_feature, out_features=18)
#         # 초기화 모델에 따라 마지막단이 (이름이) classifier가 아닐 수 있습니다.
#         # torch.nn.init.xavier_uniform_(self.model.classifier.weight)
#         # stdv = 1. / math.sqrt(self.model.classifier.weight.size(1))
#         # self.model.classifier.bias.data.uniform_(-stdv, stdv)

#     def forward(self, x):
#         x = self.model(x)
#         return x

def rand_bbox(size, lam):  # size : [Batch_size, Channel, Width, Height]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)  # 패치 크기 비율
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # 패치의 중앙 좌표 값 cx, cy
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 모서리 좌표 값
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
#     model.train()

#     t = time.time()
#     running_loss = None

#     pbar = tqdm(enumerate(train_loader), total=len(train_loader))
#     for step, (imgs, image_labels) in pbar:
#         imgs = imgs.to(device).float()
#         image_labels = image_labels.to(device).long()

#         cutmix = False
#         # cutmix 실행 될 경우
#         if np.random.random() > 0.5 and CFG['config_BETA'] > 0:
#             lam = np.random.beta(CFG['config_BETA'], CFG['config_BETA'])
#             rand_index = torch.randperm(imgs.size()[0]).to(device)
#             target_a = image_labels
#             target_b = image_labels[rand_index]
#             bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
#             imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index,
#                                                     :, bbx1:bbx2, bby1:bby2]
#             lam = 1 - ((bbx2-bbx1)*(bby2-bby1) /
#                        (imgs.size()[-1]*imgs.size()[-2]))
#             cutmix = True

#         with autocast():
#             image_preds = model(imgs)

#             if cutmix:
#                 loss = loss_fn(image_preds, target_a)*lam + \
#                     loss_fn(image_preds, target_b)*(1. - lam)
#                 cutmix = False
#             else:
#                 loss = loss_fn(image_preds, image_labels)

#             scaler.scale(loss).backward()

#             if running_loss is None:
#                 running_loss = loss.item()
#             else:
#                 running_loss = running_loss * .99 + loss.item() * .01

#             if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
#                 # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()

#                 if scheduler is not None and schd_batch_update:
#                     scheduler.step()

#             if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
#                 description = f'epoch {epoch} loss: {running_loss:.4f}'
#                 pbar.set_description(description)

#     if scheduler is not None and not schd_batch_update:
#         scheduler.step()

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, logger, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        cutmix = False
        # cutmix 실행 될 경우
        if np.random.random() > 0.5 and CFG['config_BETA'] > 0:
            lam = np.random.beta(CFG['config_BETA'], CFG['config_BETA'])
            rand_index = torch.randperm(imgs.size()[0]).to(device)
            target_a = image_labels
            target_b = image_labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index,
                                                    :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2-bbx1)*(bby2-bby1) /
                       (imgs.size()[-1]*imgs.size()[-2]))
            cutmix = True

        with autocast():
            image_preds = model(imgs)

            if cutmix:
                loss = loss_fn(image_preds, target_a)*lam + \
                    loss_fn(image_preds, target_b)*(1. - lam)
                cutmix = False
            else:
                loss = loss_fn(image_preds, image_labels)

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                pbar.set_description(description)
                logger.add_scalar("Train/loss", running_loss,
                                  epoch * len(train_loader) + step)
        if step % len(train_loader) - 1 == 0:
            img_grid = torchvision.utils.make_grid(tensor=imgs)
            logger.add_image(f'{step}_train_input_img', img_grid, step)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # output = model(input)
        image_preds_all += [torch.argmax(image_preds,
                                         1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)
        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class f1_score = {:.4f}'.format(
        f1_score(image_preds_all, image_targets_all, average='macro')))
    print(classification_report(
        valid_.class_label, np.argmax(val_preds, axis=1)))
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()

    return f1_score(image_preds_all, image_targets_all, average='macro')


def make_save_dir(path, filename):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists('{}/{}'.format(path, filename)):
        os.mkdir('{}/{}'.format(path, filename))


if __name__ == "__main__":
    seed_everything(CFG['seed'])

    # Tensorboard
    logger = SummaryWriter(log_dir='logs/{}'.format(CFG['saved_file_name']))

    make_save_dir(
        '/opt/ml/image-classification-level1-31/Jaehyun/saved_model/', CFG['saved_file_name'])
    # train = pd.read_csv('/opt/ml/input/data/train/train2.csv')
    # valid = pd.read_csv('/opt/ml/input/data/train/valid.csv')

    # raw_train = SepValidTrain().make_tmp_labeled_df()
    raw_train = pd.read_csv('/opt/ml/input/data/train/train3.csv')

    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(
        np.arange(raw_train.shape[0]), raw_train.tmp_label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # if fold > 0:
        #     break

        train_ = raw_train.loc[trn_idx, :].reset_index(drop=True)
        valid_ = raw_train.loc[val_idx, :].reset_index(drop=True)

        train = SepValidTrain().make_detailpath_N_label_df(train_)
        valid = SepValidTrain().make_detailpath_N_label_df(valid_)

        print('Training with {} started'.format(fold))
        train_loader, val_loader = prepare_dataloader(train, valid)

        device = torch.device(CFG['device'])
        model = MaskClassifier(
            CFG['model_arch'], train['class_label'].nunique(), True)
        model = model.to(device)

        scaler = GradScaler()
        optimizer = optim.Adam(
            model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)

        loss_tr = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        best_valid_f1 = 0.7

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader,
                            device, logger, scheduler=scheduler, schd_batch_update=False)
            with torch.no_grad():
                valid_f1 = valid_one_epoch(
                    epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
            folder_name = os.path.join(
                '/opt/ml/image-classification-level1-31/Jaehyun/saved_model/', CFG['saved_file_name'])
            if best_valid_f1 < valid_f1:
                torch.save(model, os.path.join(folder_name, '{}_fold_{}_{}_{}.pt'.format(
                    CFG['model_arch'], fold, epoch, np.round(valid_f1, 3))))
                best_valid_f1 = valid_f1
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
