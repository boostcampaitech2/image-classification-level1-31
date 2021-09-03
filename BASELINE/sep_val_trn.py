import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os


def sep_val_train(num_split, csv_file_path='/opt/ml/input/data/train/train.csv',
                  new_trainfile_path='/opt/ml/input/data/train/new_train.csv',
                  new_validfile_path='/opt/ml/input/data/train/new_valid.csv', seed=719):
    """[summary]
    gender와 age를 기준으로 tmp label을 부여하고 tmp label를 기준으로 StratifiedKFold
    실행하고 나온 결과물을 image file path에 따라 label이 되어있는 train.csv, valid.csv 생성
    Args:
        num_split : Kflod로 나눌 fold수
        csv_file_path : train.csv 파일 경로
        new_trainfile_path : 새로 저장할 train.csv 이름과 경로
        new_validfile_path : 새로 저장할 valid.csv 이름과 경로
    Returns:
        [type]: [description]
    """
    train_df = pd.read_csv(csv_file_path)
    label_encoder = {'female': 3, 'male': 0}

    def age_encoder(ages):
        tmp = []
        for age in ages:
            age = int(age)
            if age < 30:
                tmp.append(0)
            elif age < 60:
                tmp.append(1)
            else:
                tmp.append(2)
        return np.array(tmp)
    train_age = train_df['age'].to_numpy()
    ages = age_encoder(train_age)
    train_gender = train_df['gender'].to_numpy()
    train_gender = list(map(lambda x: label_encoder[x], train_gender))
    train_gender = np.array(train_gender)
    tmp_label = ages + train_gender
    train_df['tmp_label'] = tmp_label

    folds = StratifiedKFold(n_splits=num_split, shuffle=True, random_state=seed).split(
        np.arange(train_df.shape[0]), train_df.tmp_label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break
        train_ = train_df.loc[trn_idx, :].reset_index(drop=True)
        valid_ = train_df.loc[val_idx, :].reset_index(drop=True)
    df = train_
    paths = list(df['path'])
    img_paths = []
    for path in paths:
        img_folder_path = os.path.join(
            '/opt/ml/input/data/train/images/', path)
        imgs = os.listdir(img_folder_path)
        for img in imgs:
            if img[0] != '.':
                img_paths.append(os.path.join(img_folder_path, img))

    def dataframe2csv(df, saved_file_path):
        paths = list(df['path'])
        ids = []
        genders = []
        races = []
        ages = []
        labels = []
        img_paths = []
        for path in paths:
            img_folder_path = os.path.join(
                '/opt/ml/input/data/train/images/', path)
            imgs = os.listdir(img_folder_path)
            for img in imgs:
                if img[0] != '.':
                    img_paths.append(os.path.join(img_folder_path, img))

        for imgpath in img_paths:
            id_, gender, race, age = imgpath.split(os.sep)[-2].split('_')
            file = imgpath.split(os.sep)[-1]
            age = int(age)

            label = 0
            if gender == 'female':
                label += 3

            if 30 <= age < 60:
                label += 1
            elif 60 <= age:
                label += 2

            if 'in' in file:
                label += 6
            elif 'nor' in file:
                label += 12
            ids.append(id_)
            genders.append(gender)
            races.append(race)
            ages.append(age)
            labels.append(label)

        new_df = pd.DataFrame()
        new_df['id'] = ids
        new_df['gender'] = genders
        new_df['race'] = races
        new_df['age'] = ages
        new_df['path'] = img_paths
        new_df['class_label'] = labels
        new_df.to_csv(saved_file_path)
    dataframe2csv(train_, new_trainfile_path)
    dataframe2csv(valid_, new_validfile_path)


if __name__ == '__main__':
    sep_val_train(10)
