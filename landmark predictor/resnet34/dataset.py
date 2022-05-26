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
import torchvision.transforms.functional as F

def get_data():

    landmarks_dataframe = pd.read_csv('F:\master thesis\\face dataset\RAF\\basic\\mix_train.csv')

    return landmarks_dataframe

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, target_transfrom=None, output_shape=(-1, 2)):
        self.img_landmarks = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transfrom
        self.output_shape = output_shape

    def __len__(self):
        return len(self.img_landmarks)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_landmarks.iloc[idx, 0] + '.jpg'
        image = io.imread(img_path)
        #image = Image.open(img_path)
        #image = image.permute(1, 2, 0)
        #plt.imshow(image.numpy())
        #plt.show()
        landmarks = self.img_landmarks.iloc[idx, 1:11]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(self.output_shape)
        sample = {'image': image, 'landmarks': landmarks}
        if self.target_transform:
            sample = self.target_transform(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToPILImage(object):
    def __init__(self):
        self.transform = transforms.ToPILImage()

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        return {'image': self.transform(image), 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
      self.transform = transforms.ToTensor()
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = self.transform(image)
        return {'image': image,
                'landmarks':torch.from_numpy(landmarks)}

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        w, h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        trnsf = transforms.Resize((self.output_size, self.output_size))
        img = trnsf(image)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]
        landmarks = landmarks * [self.output_size / w, self.output_size / h]
        return {'image': img, 'landmarks': landmarks}

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
    
        img, landmarks = sample['image'], sample['landmarks']
        #print(img.shape)
        w, h = img.size
        if torch.rand(1) < self.p:
            img2 = F.hflip(img)
            #landmarks2 = landmarks.copy()
            landmarks[:,0] = abs(landmarks[:,0] - w)
            return {'image': img2, 'landmarks': landmarks}

        return {'image': img, 'landmarks': landmarks}


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        
        img, landmarks = sample['image'], sample['landmarks']
        w, h = img.size
        if torch.rand(1) < self.p:
            img = F.vflip(img)
            landmarks[:, 1] = abs(landmarks[:, 1] - h)

        return {'image': img, 'landmarks': landmarks}



landmarks_dataframe = get_data()

