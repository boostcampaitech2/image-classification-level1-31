from torch.utils.data import Dataset
import cv2
import torch


class MaskDataset(Dataset):
    def __init__(
        self, df, transforms=None, output_label=True
    ):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.output_label = output_label
        if output_label:
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
