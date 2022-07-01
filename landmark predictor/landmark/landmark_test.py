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
import landmark_dataset as datasets
import argparse


transfrom_list = [datasets.Rescale(227), datasets.ToTensor()]
img_to_tensor = T.Compose(transfrom_list)


def make_model(predictor):
    resnet = models.resnet34(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, 10)
    #resnet.load_state_dict(torch.load('E:\\microsoft dataframe\\microsoft_resnet.pt'))
    #resnet.load_state_dict(torch.load('E:\\3d_morphable_model_199\\resnet34\\balanced_resnet.pt'))
    resnet.load_state_dict(torch.load(predictor))
    resnet.cuda()

    return resnet

def inference(resnetmodel, imgpath, gth_ldmk):
    with torch.no_grad():
        resnetmodel.eval()
        img = Image.open(imgpath)
        sample = {'image': img, 'landmarks': gth_ldmk}
        tensor = img_to_tensor(sample)

        tensor = tensor['image'].resize_(1, 3, 227, 227)
        tensor = tensor.cuda()

        result = resnetmodel(Variable(tensor))
        result_npy = result.data.cpu().numpy()

    return result_npy[0]

def mean_distance(df):
    a = np.sqrt(np.square(df['Ax'] - df['AxPre']) + np.square(df['Ay'] - df['AyPre']))
    b = np.sqrt(np.square(df['Fx'] - df['FxPre']) + np.square(df['Fy'] - df['FyPre']))
    c = np.sqrt(np.square(df['Qx'] - df['QxPre']) + np.square(df['Qy'] - df['QyPre']))
    d = np.sqrt(np.square(df['Ux'] - df['UxPre']) + np.square(df['Uy'] - df['UyPre']))
    e = np.sqrt(np.square(df['Yx'] - df['YxPre']) + np.square(df['Yy'] - df['YyPre']))
    df['mean_dis'] = (a+b+c+d+e) / 5.

    return df

def mean_absolute_error(df):
    a = np.absolute(df['Ax'] - df['AxPre']) + np.absolute(df['Ay'] - df['AyPre'])
    b = np.absolute(df['Fx'] - df['FxPre']) + np.absolute(df['Fy'] - df['FyPre'])
    c = np.absolute(df['Qx'] - df['QxPre']) + np.absolute(df['Qy'] - df['QyPre'])
    d = np.absolute(df['Ux'] - df['UxPre']) + np.absolute(df['Uy'] - df['UyPre'])
    e = np.absolute(df['Yx'] - df['YxPre']) + np.absolute(df['Yy'] - df['YyPre'])
    df['MAE'] = (a+b+c+d+e) / 10.

    return df

def show_landmarks(image, landmarks_gt, landmarks_pred, ax_new):
    """Show image with landmarks"""
    ax_new.imshow(image)
    ax_new.scatter(landmarks_gt[:, 0], landmarks_gt[:, 1], s=20, marker='.', c=['g'])
    ax_new.scatter(landmarks_pred[:, 0], landmarks_pred[:, 1], s=20, marker='.', c=['r'])

if __name__=="__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-t', "--testfile", required=True, help= "test file path")
    ap.add_argument('-i', "--imagefile", required=True, help="image file path")
    ap.add_argument('-p', "--predictor", required=True, help="trained predictor")
    args = vars(ap.parse_args())

    landmarks = ['Image', 'AxPre', 'AyPre', 'FxPre', 'FyPre', 'QxPre', 'QyPre', 'UxPre', 'UyPre', 'YxPre', 'YyPre']
    model = make_model(args["predictor"])
    was_training = model.training
    races = ['asian', 'black', 'white']
    for race in races:
        pre_gender = pd.DataFrame(
            columns=['Image', 'AxPre', 'AyPre', 'FxPre', 'FyPre', 'QxPre', 'QyPre', 'UxPre', 'UyPre', 'YxPre', 'YyPre'])
        filepath = args["testfile"] + race + '_test.csv'
        imgpath = args["imagefile"]#'E:\\facedataset2\\'
        df = pd.read_csv(filepath).iloc[:5]

        images_so_far = 0
        print(df.head())
        for i in df['Image']:

            gth = df[df['Image'] == i][['Ax', 'Ay','Fx', 'Fy','Qx', 'Qy','Ux', 'Uy','Yx', 'Yy']]
            gth = np.array(gth.values[0]).reshape(-1,2)
            #print(gth)
            img_path = imgpath + i
            img = Image.open(img_path)
            #w, h = img.size
            #print(w, h)
            preds = inference(model, img_path, gth).reshape(-1, 2)
            #print(preds)
            #if w > h:
            #    preds = preds * np.array([h / 224, h / 224])
            #else:
            #    preds = preds * np.array([w / 224, w / 224])
            #preds = preds * np.array([w / 227, h / 227])
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
                plt.savefig('E:\\3d_morphable_model_199\\resnet34/balanced_test_7/' + race + '_example.png', bbox_inches='tight')
                plt.show()


            ldmks = preds.flatten().tolist()
            ldmks.insert(0, i)
            s = dict(zip(landmarks, ldmks))
            pre_gender = pre_gender.concat(s, ignore_index=True)

        #print(pre_gender.head())
        #df = df.merge(pre_gender, left_on='Image', right_on='Image')
        #df.set_index('Image', inplace=True)
        #df = mean_distance(df)
        #df = mean_absolute_error(df)
        #df.to_csv('E:\\3d_morphable_model_199\\resnet34\\white_test_7\\' + race + '_distance.csv')
        #print(df.head())