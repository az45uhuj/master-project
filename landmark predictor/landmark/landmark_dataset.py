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

def get_data():

    landmarks_dataframe = pd.read_csv('/media/qi/Elements/3d_morphable_model_199/balanced_train_ldmks.csv').iloc[:100]



    #asian = pd.read_csv('E:\\facedataset\\asian\\landmarks.csv').iloc[:2500]
    #black = pd.read_csv('E:\\facedataset\\black\\landmarks.csv').iloc[:2500]
    #caucasian = pd.read_csv('E:\\facedataset\\caucasian\\landmarks.csv').iloc[:2500]
    #hispanic = pd.read_csv('E:\\facedataset\\hisp\\landmarks.csv').iloc[:2500]
    #landmarks_dataframe = pd.read_csv('E:\\facedataset\\hisp\\landmarks.csv').iloc[:10000]
    #landmarks_dataframe = pd.concat([asian, black, caucasian, hispanic])

    '''
    df = pd.read_csv('E:\\facedataset\\hispanic_ldmks.csv')
    landmarks_dataframe = df[['Image', 'Ax', 'Ay', 'Bx', 'By', 'Cx', 'Cy', 'Dx', 'Dy', 'Ex', 'Ey']]
    landmarks_dataframe = (landmarks_dataframe.astype(int))
    path1 = ['hisp\\img\\'] * 100000
    s_path1 = pd.Series(path1)
    path2 = ['\\'] * 100000
    s_path2 = pd.Series(path2)
    path3 = ['_0.png'] * 100000
    s_path3 = pd.Series(path3)
    landmarks_dataframe['Image'] = s_path1.str.cat(landmarks_dataframe['Image'].astype(str)).str.cat(s_path2)\
        .str.cat(landmarks_dataframe['Image'].astype(str)).str.cat(s_path3)
    landmarks_dataframe.set_index('Image', inplace=True)
    landmarks_dataframe.to_csv('E:\\facedataset\\hisp\landmarks.csv')
    print(landmarks_dataframe.head())
    '''

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
        img_path = self.img_dir + self.img_landmarks.iloc[idx, 0]
        image = io.imread(img_path)
        #image = Image.open(img_path)
        #image = image.permute(1, 2, 0)
        #plt.imshow(image.numpy())
        #plt.show()
        landmarks = self.img_landmarks.iloc[idx, 1:]
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

#file_path = 'F:\master thesis\\face dataset\Fairface\\'
landmarks_dataframe = get_data()
print(landmarks_dataframe.head())
print(len(landmarks_dataframe))
#race_dataset = CustomImageDataset(race_dataframe, file_path)
#race_dataset.__getitem__(2)