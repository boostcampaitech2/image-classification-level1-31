import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os


class SepValidTrain():
    def __init__(self, csv_file_path='/opt/ml/input/data/train/train.csv', seed=719) -> None:
        self.csv_file_path = csv_file_path
        self.seed = seed

    def make_tmp_labeled_df(self):
        train_df = pd.read_csv(self.csv_file_path)
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

        return train_df

    def make_detailpath_N_label_df(self, raw_df):
        paths = list(raw_df['path'])
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
        return new_df


if __name__ == '__main__':
    sep = SepValidTrain()
    raw_label_data = sep.make_tmp_labeled_df()
    split = StratifiedKFold(7, shuffle=True, random_state=718)
    folds = split.split(
        np.arange(raw_label_data.shape[0]), raw_label_data.tmp_label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        train_ = raw_label_data.loc[trn_idx, :]
        train_.to_csv('/opt/ml/input/data/train/train3.csv')

        test_ = raw_label_data.loc[val_idx, :]
        test_df = sep.make_detailpath_N_label_df(test_)
        test_df.to_csv('/opt/ml/input/data/train/test.csv')
