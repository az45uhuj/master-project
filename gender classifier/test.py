import torch
from torchvision import models
from PIL import Image
import numpy as np
from torchvision.io import read_image
import torchvision.transforms as T
from torch.autograd import Variable
import pandas as pd


transfrom_list = [T.ToTensor()]
img_to_tensor = T.Compose(transfrom_list)

def make_model():
    vgg = models.resnet18(pretrained=True)
    vgg.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                          torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 2))

    #vgg.to(device)
    vgg.load_state_dict(torch.load('E:\\faireface_gendermodel\\vgg model3\\asian_vgg16.pt')) # load the trained model
    vgg.cuda()

    return resnet

def inference(vggmodel, imgpath):
    vggmodel.eval()
    img = Image.open(imgpath)
    img=img.resize((224, 224))
    tensor = img_to_tensor(img)

    tensor = tensor.resize_(1, 3, 224, 224)
    tensor = tensor.cuda()

    result = vggmodel(Variable(tensor))
    result_npy = result.data.cpu().numpy()
    max_index = np.argmax(result_npy[0])

    return max_index

if __name__=="__main__":
    d = {}
    model = make_model()
    races = ['asian', 'black', 'hispanic', 'indian', 'middle_eastern', 'white']
    for race in races:
        filepath = 'F:\master thesis\\face dataset\Fairface\\train_dataset\\' + race + '_test.csv'
        imgpath = 'F:\master thesis\\face dataset\Fairface\\'
        df = pd.read_csv(filepath)[['file', 'gender_label', 'age_label']]

        print(df.head())
        for i in df['file']:
            img_path = imgpath + i
            predict = inference(model, img_path)
            d[i] = predict
        pre_gender = pd.DataFrame(d.items(), columns=['Image', 'Pre_Gender'])
        df = df.merge(pre_gender, left_on='file', right_on='Image')
        df.set_index('Image', inplace=True)
        df['error'] = abs(df['gender_label'] - df['Pre_Gender'])
        df = df.drop(['file'], axis=1)
        df.to_csv('E:\\faireface_agemodel\\resnet model3\\mix_test\\' + race + '_pre_gender.csv')
        print(df.head())
