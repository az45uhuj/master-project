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

    gender_dataframe = pd.read_csv('gender_train.csv')

    return gender_dataframe

gender_dataframe = get_data()
