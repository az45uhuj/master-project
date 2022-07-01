import torch
#import VGG16
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

    resnet = models.resnet34(pretrained=True)
    resnet.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(resnet.fc.in_features, 9)
    )

    resnet.load_state_dict(torch.load('/media/qi/Elements/master project/age_model/resnet model3/asian_resnet34_2.pt'))
    resnet.cuda()

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
    races = ['asian']
    #races = ['asian', 'black', 'hispanic', 'indian', 'middle_eastern', 'white']
    for race in races:
        filepath = '/media/qi/Elements/windows/master thesis/face dataset/Fairface/new_age_train_and_test_dataset2/' + race + '_test.csv'
        imgpath = '/media/qi/Elements/windows/master thesis/face dataset/Fairface/'
        df = pd.read_csv(filepath)[['file', 'gender_label', 'age_label']]

        print(df.head())
        for i in df['file']:
            img_path = imgpath + i
            predict = inference(model, img_path)
            d[i] = predict
        pre_gender = pd.DataFrame(d.items(), columns=['Image', 'Pre_Age'])
        df = df.merge(pre_gender, left_on='file', right_on='Image')
        df.set_index('Image', inplace=True)
        df['error'] = abs(df['age_label'] - df['Pre_Age'])
        df = df.drop(['file'], axis=1)
        df.to_csv('/media/qi/Elements/master project/age_model/asian2/' + race + '_pre_age.csv')
        print(df.head())
