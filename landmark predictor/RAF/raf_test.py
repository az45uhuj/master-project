import torch
from torchvision import models
from PIL import Image
import numpy as np
from torchvision.io import read_image
import torchvision.transforms as T
from torch.autograd import Variable
import pandas as pd
import math
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import cv2
import raf_landmark_dataset as dataset


transfrom_list = [dataset.Rescale(224), dataset.ToTensor()]
img_to_tensor = T.Compose(transfrom_list)


def make_model():
    resnet = models.resnet34(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, 10)
    #resnet.load_state_dict(torch.load('E:\\microsoft dataframe\\microsoft_resnet.pt'))
    resnet.load_state_dict(torch.load('/media/qi/Elements/master project/RAF_landmarks/asian_resnet34.pt'))
    resnet.cuda()

    return resnet

def inference(resnetmodel, imgpath, gth_ldmk):
    with torch.no_grad():
        resnetmodel.eval()
        img = Image.open(imgpath)
        sample = {'image': img, 'landmarks': gth_ldmk}
        tensor = img_to_tensor(sample)

        tensor = tensor['image'].resize_(1, 3, 224, 224)
        tensor = tensor.cuda()

        result = resnetmodel(Variable(tensor))
        result_npy = result.data.cpu().numpy()

    return result_npy[0]

def mean_distance(df):
    a = np.sqrt(np.square(df['left_eye_centerx'] - df['p_left_eye_centerx']) + np.square(df['left_eye_centery'] - df['p_left_eye_centery']))
    b = np.sqrt(np.square(df['right_eye_centerx'] - df['p_right_eye_centerx']) + np.square(df['right_eye_centery'] - df['p_right_eye_centery']))
    c = np.sqrt(np.square(df['nose_tipx'] - df['p_nose_tipx']) + np.square(df['nose_tipy'] - df['p_nose_tipy']))
    d = np.sqrt(np.square(df['mouth_left_cornerx'] - df['p_mouth_left_cornerx']) + np.square(df['mouth_left_cornery'] - df['p_mouth_left_cornery']))
    e = np.sqrt(np.square(df['mouth_right_cornerx'] - df['p_mouth_right_cornerx']) + np.square(df['mouth_right_cornery'] - df['p_mouth_right_cornery']))
    df['mean_dis'] = (a+b+c+d+e) / 5.

    return df

def show_landmarks(image, landmarks_gt, landmarks_pred, ax_new):
    """Show image with landmarks"""
    ax_new.imshow(image)
    ax_new.scatter(landmarks_gt[:, 0], landmarks_gt[:, 1], s=20, marker='.', c=['g'])
    ax_new.scatter(landmarks_pred[:, 0], landmarks_pred[:, 1], s=20, marker='.', c=['r'])

if __name__=="__main__":
    landmarks = ['Image', 'p_left_eye_centerx', 'p_left_eye_centery', 'p_right_eye_centerx', 'p_right_eye_centery', 'p_nose_tipx',
                     'p_nose_tipy', 'p_mouth_left_cornerx', 'p_mouth_left_cornery','p_mouth_right_cornerx', 'p_mouth_right_cornery']

    model = make_model()
    was_training = model.training
    races = ['asian', 'black', 'caucasian']
    for race in races:
        pre_gender = pd.DataFrame(
            columns=['Image','p_left_eye_centerx', 'p_left_eye_centery', 'p_right_eye_centerx', 'p_right_eye_centery', 'p_nose_tipx',
                     'p_nose_tipy', 'p_mouth_left_cornerx', 'p_mouth_left_cornery','p_mouth_right_cornerx', 'p_mouth_right_cornery'])
        filepath = '/media/qi/Elements/windows/master thesis/face dataset/RAF/basic/' + race + '_test.csv'
        imgpath = '/media/qi/Elements/windows/master thesis/face dataset/RAF/basic/Image/original/'
        df = pd.read_csv(filepath).iloc[[0,2,4,66,47]]

        images_so_far = 0
        print(df.head())
        for i in df['Image']:

            gth = df[df['Image'] == i][['left_eye_centerx', 'left_eye_centery', 'right_eye_centerx', 'right_eye_centery', 'nose_tipx',
                 'nose_tipy', 'mouth_left_cornerx', 'mouth_left_cornery','mouth_right_cornerx', 'mouth_right_cornery']]
            gth = np.array(gth.values[0]).reshape(-1,2)
            #print(gth)
            img_path = imgpath + i + '.jpg'
            img = Image.open(img_path)
            w, h = img.size
            #print(w, h)
            preds = inference(model, img_path, gth).reshape(-1, 2)
            #print(preds)
            #if w > h:
            #    preds = preds * np.array([h / 224, h / 224])
            #else:
            #    preds = preds * np.array([w / 224, w / 224])
            preds = preds * np.array([w / 224, h / 224])
            #print(preds)


            images_so_far += 1
            img = Image.open(img_path)
            ax = plt.subplot(1, 5, images_so_far)
            ax.imshow(img)
            ax.set_title('{:.3f}'.format(((gth- preds) ** 2).mean().item()))
            ax.axis('off')
            show_landmarks(img, gth, np.array(preds), ax)
            if images_so_far == 5:
                model.train(mode=was_training)
                #plt.savefig('E:\RAF_landmarks\\mix_test/' + race + '_example3.png', bbox_inches='tight')
                plt.show()


            ldmks = preds.flatten().tolist()
            ldmks.insert(0, i)
            s = dict(zip(landmarks, ldmks))
            pre_gender = pre_gender.append(s,ignore_index=True)


        #print(pre_gender.head())
        #df = df.merge(pre_gender, left_on='Image', right_on='Image')
        #df.set_index('Image', inplace=True)
        #df = mean_distance(df)
        #df.to_csv('E:\RAF_landmarks\\mix_test/' + race + '_distance.csv')
        #print(df.head())