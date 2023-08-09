"""
This script is used to allocate the dataset into training, validation, and testing groups.
"""
import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time

class PorcineDataset(Dataset):
    """Dataset for PSEA, PSEC, PPC, PSTRAIN"""

    def __init__(self, data_path, transform=None):#, is_train, train_split = 0.9, random_seed = 42, target_transform = None, num_classes = None):
        super(PorcineDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform

        self.image_list = []
        for image_name in os.listdir(data_path):
            image_path = os.path.join(data_path, image_name)
            self.image_list.append(dict(
                image_path=image_path,
                image_name=image_name,
            ))

        self.img_idxes = np.arange(0, len(self.image_list))


    def __len__(self):
        return len(self.img_idxes)

    def __getitem__(self, index):

        img_idx = self.img_idxes[index]
        img_info = self.image_list[img_idx]

        data = sio.loadmat(img_info['image_path'])
        filename = img_info['image_name']
        img = data['img_resampled']  # img_resampled
        myocardium1 = data['myo_begin']
        myocardium2 = data['myo_end']
        if self.transform:
            img = self.transform(img)
            myocardium1 = self.transform(myocardium1)
            myocardium2 = self.transform(myocardium2)

        return img, myocardium1, myocardium2, filename

    def get_number_of_samples(self):
        return self.__len__()