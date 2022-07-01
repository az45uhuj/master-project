import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms, utils
from skimage import io, transform
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

def get_data():

    landmarks_dataframe = pd.read_csv('/media/qi/Elements/windows/master thesis/face dataset/Fairface/new_age_train_and_test_dataset2/asian_train.csv').iloc[:100]

    return landmarks_dataframe

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform):
        self.img_age = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_age)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_age.iloc[idx, 0]
        image = io.imread(img_path)
        #image = Image.open(img_path)
        #image = image.permute(1, 2, 0)
        #plt.imshow(image.numpy())
        #plt.show()
        age = np.array(self.img_age.iloc[idx, 6])
        sample = {'image': image, 'age': age}
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class ToPILImage(object):
    def __init__(self):
        self.transform = transforms.ToPILImage()

    def __call__(self, image):
        return self.transform(image)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
      self.transform = transforms.ToTensor()

    def __call__(self, image):
        image = self.transform(image)
        return image


class RandomHorizontalFlip(object):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        img, landmarks = sample['image'], sample['landmarks']
        w, h = img.size
        if torch.rand(1) < self.p:
            img =  F.hflip(img)
            landmarks[0] = w - landmarks[0]

        return {'image': img, 'landmarks': landmarks}


class RandomVerticalFlip(object):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        img, landmarks = sample['image'], sample['landmarks']
        w, h = img.size
        if torch.rand(1) < self.p:
            img = F.vflip(img)
            landmarks[1] = h - landmarks[1]

        return {'image': img, 'landmarks': landmarks}


#file_path = 'F:\master thesis\\face dataset\Fairface\\'
race_dataframe = get_data()
print(race_dataframe.head())
print(len(race_dataframe))
#race_dataset = CustomImageDataset(race_dataframe, file_path)
#race_dataset.__getitem__(2)