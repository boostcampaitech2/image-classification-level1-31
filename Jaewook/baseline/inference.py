import argparse
import os
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset, MaskSplitByProfileDataset

from sklearn.metrics import f1_score, classification_report

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, '6_0.08822816647589207.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_models(saved_model_path, num_classes, device):
    models = []
    exps = ['exp7', 'exp8', 'exp10']
    for exp, model in zip(exps, args.models):
        print(model)
        model_cls = getattr(import_module("model"), model)
        model = model_cls(
            num_classes=num_classes
        )
        if exp == 'exp10':
            model_path = os.path.join(saved_model_path, exp, 'best.pth')
        else:
            model_path = os.path.join(saved_model_path, exp, 'best.pt')


        model.load_state_dict(torch.load(model_path, map_location=device))
        models.append(model)

    return models

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')

@torch.no_grad()
def ensemble_inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskSplitByProfileDataset.num_classes  # 18
    models = load_models(model_dir, num_classes, device)

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = MaskSplitByProfileDataset('../../../input/data/train/images')
    test_dataset = TestDataset(img_paths, args.resize)
    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=test_dataset.mean,
        std=test_dataset.std,
    )
    dataset.set_transform(transform)
    _, val_dataset = dataset.split_dataset()
    test_dataset.set_transform(transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    # val loop
    val_f1_preds = []
    val_f1_labels = []
    for model in models:
        model.to(device)
        val_preds = []
        val_labels = []
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                val_preds.extend(preds.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())
        
        val_f1_preds.append(val_preds)
    
    val_f1_preds = np.mean(val_f1_preds, axis=0)
    val_f1_preds = np.around(val_f1_preds)
    val_f1_labels = val_labels

    val_f1_score = f1_score(val_f1_labels, val_f1_preds, average='macro')
    print(f'val_f1_score: {val_f1_score}')
    print(classification_report(val_f1_labels, val_f1_preds))

    # test loop
    test_final_preds = []
    for model in models:
        model.to(device)
        test_preds = []
        with torch.no_grad():
            print("Calculating test results...")
            model.eval()
            for test_batch in test_loader:
                inputs = test_batch
                inputs = inputs.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                test_preds.extend(preds.detach().cpu().numpy())

        test_final_preds.append(test_preds)
    
    # print(f'test_f1_preds: {test_final_preds}')
    test_final_preds = np.mean(test_final_preds, axis=0)
    # print(f'test_f1_preds: {test_final_preds}')
    test_final_preds = np.around(test_final_preds)
    test_final_preds = list(map(int, test_final_preds))

    info['ans'] = test_final_preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(384, 384), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--models', type=list, default=['NGModel', 'JHModel', 'EffB4'], help='model type (default: BaseModel)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='augmentation type (default: BaseAugmentation)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
    # ensemble_inference(data_dir, model_dir, output_dir, args)