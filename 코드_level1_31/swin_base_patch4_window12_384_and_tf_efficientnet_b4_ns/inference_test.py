import torch

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

# from torchvision import transforms
from torch.utils.data import DataLoader

import random
import pandas as pd
import numpy as np

from tqdm import tqdm
import os

from sklearn.metrics import f1_score, log_loss
from sklearn.metrics import classification_report
from data_utils.datasets import MaskDataset
from model.models import *
from model.loss import *
model_class = {'swin_base_patch4_window12_384': 'Transformer',
               'tf_efficientnet_b4_ns':  'EfficientNet',
               }

CFG = {
    'seed': 719,
    # 'model_arch': 'vit_base_patch16_384',
    'model_arch': 'swin_base_patch4_window12_384',
    'epochs': 10,
    'valid_bs': 32,
    'num_workers': 4,
    # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'save_model_path': '../data',  # 저장된 모델들의 폴더 경로
    'saved_file_name': 'ensemble',
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
    저장한 모델 중 f1 score가 가장 높은 모델부터 num개 list로 가져옴
    Args:
        path ([str]): 모델 저장 폴더 위치
        num ([int): 가져오고 싶은 모델 개수
    Return:
        path에 있는 파일명들 list ([list])
    """
    tmp = {}
    saved_model_path = path
    filelist = os.listdir(saved_model_path)
    for file in filelist:
        if file[0] != '.' and file != 'log.txt':
            f1_score = file.split('_')[-1]
            if f1_score.split('.')[-1] == 'pt':
                tmp[float(f1_score[:-3])] = file
            elif f1_score.split('.')[-1] == 'pth':
                tmp[float(f1_score[:-4])] = file
    selectmodel = sorted(list(tmp.keys()), reverse=True)[:num]
    return [tmp[model] for model in selectmodel]


if __name__ == "__main__":
    seed_everything(CFG['seed'])

    f_name = '/opt/ml/input/data/eval/images'  # inference image folder
    test_df = pd.read_csv("/opt/ml/input/data/eval/info.csv")
    test_df['path'] = test_df['ImageID'].apply(lambda x: "/".join([f_name, x]))

    print('Inference started')

    valid_ = pd.read_csv('/opt/ml/input/data/train/test.csv')

    valid_ds = MaskDataset(
        valid_, transforms=get_inference_transforms(), output_label=False)

    test_ds = MaskDataset(
        test_df, transforms=get_inference_transforms(), output_label=False)

    val_loader = DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    tst_loader = DataLoader(
        test_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    model_folder = os.path.join(
        CFG['save_model_path'], CFG['saved_file_name'])  # 모델이 저장된 폴더
    # 사용할 모델 리스트
    models = find_best_model(model_folder, CFG['ensemble_num'])

    device = torch.device(CFG['device'])

    val_preds = []
    tst_preds = []

    for i, model_version in enumerate(models):

        # 저장하는 모델의 이름을 통일하지 않아 임시적으로 model load 구성
        if 'vgg19' in model_version:
            model = MyModel_na(18)
            model.load_state_dict(torch.load(model_folder+"/"+model_version))

        elif 'best' in model_version:
            model = TestModel(18)
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
            val_preds += [inference_one_epoch(model, val_loader, device)]
            tst_preds += [inference_one_epoch(model, tst_loader, device)]

    val_preds = np.mean(val_preds, axis=0)
    tst_preds = np.mean(tst_preds, axis=0)

    print('validation loss = {:.5f}'.format(
        log_loss(valid_.class_label.values, val_preds)))
    print('validation f1-score = {:.5f}'.format(
        f1_score(valid_.class_label.values, np.argmax(val_preds, axis=1), average='macro')))
    print(classification_report(
        valid_.class_label, np.argmax(val_preds, axis=1)))

    submission = pd.read_csv("/opt/ml/input/data/eval/info.csv")
    submission['ans'] = np.argmax(tst_preds, axis=1)
    submission.to_csv(
        os.path.join("../submmision_files", "{}.csv".format(CFG['saved_file_name'])), index=False)
