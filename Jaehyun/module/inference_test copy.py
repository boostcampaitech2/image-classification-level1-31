import torch

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import random
import pandas as pd
import numpy as np

from tqdm import tqdm
import timm
import os

from sklearn.metrics import f1_score, log_loss
from sklearn.metrics import classification_report
from data_utils.datasets import MaskDataset
from data_utils.data_loaders import MaskDataLoader
from model.models import *
from model.loss import *
model_class = {'swin_base_patch4_window12_384': 'Transformer',
               'tf_efficientnet_b4_ns':  'EfficientNet',
               }

CFG = {
    'fold_num': 10,
    'seed': 719,
    # 'model_arch': 'vit_base_patch16_384',
    'model_arch': 'swin_base_patch4_window12_384',
    'img_size': 384,
    'epochs': 10,
    'train_bs': 16,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'accum_iter': 2,
    'verbose_step': 1,
    'device': 'cuda:0',
    'save_model_path': '/opt/ml/image-classification-level1-31/Jaehyun/saved_model',
    'saved_file_name': 'last',
    'ensemble_num': 10
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_inference_transforms():
    return Compose([
        A.Resize(height=CFG['img_size'], width=CFG['img_size']),
        A.Normalize(),
        ToTensorV2()
    ])


def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()

        image_preds = model(imgs)  # output = model(input)
        image_preds_all += [(image_preds.detach().cpu())]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def find_best_model(path, num):
    """[summary]
    ????????? ?????? ??? f1 score??? ?????? ?????? ???????????? num??? list??? ?????????
    Args:
        path ([str]): ?????? ?????? ?????? ??????
        num ([int): ???????????? ?????? ?????? ??????
    Return:
        path??? ?????? ???????????? list ([list])
    """
    tmp = []
    saved_model_path = path
    filelist = os.listdir(saved_model_path)
    for file in filelist:
        if file[0] != '.':
            tmp.append(file)
    return tmp


if __name__ == "__main__":
    seed_everything(CFG['seed'])

    f_name = '/opt/ml/input/data/eval/images'  # inference image folder
    test_df = pd.read_csv("/opt/ml/input/data/eval/info.csv")
    test_df['path'] = test_df['ImageID'].apply(lambda x: "/".join([f_name, x]))

    print('Inference started')

    # valid_ = pd.read_csv('/opt/ml/input/data/train/test.csv')

    # valid_ds = MaskDataset(
    #     valid_, transforms=get_inference_transforms(), output_label=False)

    test_ds = MaskDataset(
        test_df, transforms=get_inference_transforms(), output_label=False)

    # val_loader = DataLoader(
    #     valid_ds,
    #     batch_size=CFG['valid_bs'],
    #     num_workers=CFG['num_workers'],
    #     shuffle=False,
    #     pin_memory=False,
    # )

    tst_loader = DataLoader(
        test_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    model_folder = os.path.join(
        CFG['save_model_path'], CFG['saved_file_name'])  # ?????? ?????? ??????
    # ????????? ?????? ?????????
    models = find_best_model(model_folder, CFG['ensemble_num'])
    # models = ['swin_base_patch4_window12_384_fold_0_3_0.798.pt',
    #           'swin_base_patch4_window12_384_fold_1_4_0.828.pt',
    #           'swin_base_patch4_window12_384_fold_2_0_0.807.pt',
    #           'swin_base_patch4_window12_384_fold_3_7_0.838.pt',
    #           'swin_base_patch4_window12_384_fold_4_5_0.851.pt'
    #           ]

    device = torch.device(CFG['device'])

    # val_preds = []
    tst_preds = []

    for i, model_version in enumerate(models):

        # ???????????? ????????? ????????? ???????????? ?????? ??????????????? model load ??????
        if model_version == '4vgg19_fold_3_0.804.pt':
            model = MyModel_na(18)
            model.load_state_dict(torch.load(model_folder+"/"+model_version))

        elif model_version == 'best_0.8.pth':
            model = TestModel(18)
            model.load_state_dict(torch.load(model_folder+"/"+model_version))

        elif model_version == 'vit_small_r26_s32_384_fold_9_46_0':
            model = MaskClassifier_dongjin(
                model_arch="vit_small_r26_s32_384", n_class=18)
            model.load_state_dict(torch.load(model_folder+"/"+model_version))

        # elif model_version == 'inception.pt':
        #     model = Mymodel()
        #     model.load_state_dict(torch.load(model_folder+"/"+model_version))

        elif model_version == 'resnet_best.pth':
            model = resnet(n_class=18)
            model.load_state_dict(torch.load(model_folder+"/"+model_version))

        else:
            try:
                model = MaskClassifier_efficient(
                    '_'.join(model_version.split('_')[:-4]), 18)
                model.load_state_dict(torch.load(
                    model_folder+"/"+model_version))
            except:
                try:
                    model = MaskClassifier_transformer(
                        '_'.join(model_version.split('_')[:-4]), 18)
                    model.load_state_dict(torch.load(
                        model_folder+"/"+model_version))
                except:
                    try:
                        model = MaskClassifier_custom_transformer(
                            '_'.join(model_version.split('_')[:-4]), 18)
                        model.load_state_dict(torch.load(
                            model_folder+"/"+model_version))
                    except:
                        try:
                            model = MaskClassifier_custom_transformer2(
                                '_'.join(model_version.split('_')[:-4]), 18)
                            model.load_state_dict(torch.load(
                                model_folder+"/"+model_version))
                        except:
                            try:
                                model = MaskClassifier(CFG['model_arch'], 18)
                                model.load_state_dict(torch.load(
                                    model_folder+"/"+model_version))
                            except:
                                model = torch.load(
                                    model_folder+"/"+model_version)
        model = model.to(device)
        with torch.no_grad():
            # val_preds += [inference_one_epoch(model, val_loader, device)]
            tst_preds += [inference_one_epoch(model, tst_loader, device)]

    # val_preds = np.mean(val_preds, axis=0)
    tst_preds = np.mean(tst_preds, axis=0)

    # print('validation loss = {:.5f}'.format(
    #     log_loss(valid_.class_label.values, val_preds)))
    # print('validation f1-score = {:.5f}'.format(
    #     f1_score(valid_.class_label.values, np.argmax(val_preds, axis=1), average='macro')))
    # print(classification_report(
    #     valid_.class_label, np.argmax(val_preds, axis=1)))

    submission = pd.read_csv("/opt/ml/input/data/eval/info.csv")
    submission['ans'] = np.argmax(tst_preds, axis=1)
    submission.to_csv(
        os.path.join("/opt/ml/image-classification-level1-31/Jaehyun/submission_files", "{}.csv".format(CFG['saved_file_name'])), index=False)
