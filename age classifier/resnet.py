import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torchvision import models
from torch.utils.data import DataLoader
import age_dataset


#train_val, test = train_test_split(datasets.landmarks_dataframe, test_size=0.2, random_state=42)
train, val = train_test_split(age_dataset.race_dataframe, test_size=0.2, random_state=42)

transform_objs = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(degrees=(0, 180)),
                                     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), transforms.ToTensor()])


img_dir = '/media/qi/Elements/windows/master thesis/face dataset/Fairface/'
train_dataset = age_dataset.CustomImageDataset(train, img_dir, transform_objs)
val_dataset = age_dataset.CustomImageDataset(val,  img_dir, transform_objs)
#test_dataset = datasets.CustomImageDataset(test,  img_dir, transform_objs)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)
#test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

resnet = models.resnet34(pretrained=True)

# freeze convolution weights
for param in resnet.parameters():
   param.requires_grad = True

#num_ftrs = resnet.fc.in_features
#resnet.fc = torch.nn.Linear(num_ftrs, 5)
resnet.fc = nn.Sequential(
     nn.Dropout(0.5),
     nn.Linear(resnet.fc.in_features, 5)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# optimizer
optimizer = optim.SGD(resnet.parameters(), lr=0.0001, momentum=0.9)
#optimizer = optim.Adam(resnet.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# loss function
criterion = nn.CrossEntropyLoss()

# validation function
def validate(model, val_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for int, data in enumerate(val_dataloader):
        data, target = data['image'].to(device), data['age'].to(device)
        output = model(data)
        loss = criterion(output, target)

        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()

    val_loss = val_running_loss / len(val_dataloader.dataset)
    val_accuracy = 100. * val_running_correct / len(val_dataloader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.2f}')

    return val_loss, val_accuracy

# training function
def fit(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data['image'].to(device), data['age'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct / len(train_dataloader.dataset)

    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')

    return train_loss, train_accuracy

def train():
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    best_loss = 1000
    #best_model_wts = copy.deepcopy(vgg16.state_dict())
    start = time.time()
    for epoch in range(50):
        train_epoch_loss, train_epoch_accuracy = fit(resnet, train_loader)
        val_epoch_loss, val_epoch_accuracy = validate(resnet, val_loader)
        #scheduler.step()
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            #torch.save(resnet.state_dict(), 'E:\\faireface_agemodel\\resnet model3\\' +'mix_resnet34.pt')
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
    end = time.time()
    print((end - start) / 60, 'minutes')
    return train_accuracy, train_loss, val_accuracy, val_loss

def visulize(trainAcc, valAcc, trainLoss, valLoss):
    # plot accuracy trend
    plt.figure(figsize=(10, 7))
    plt.plot(trainAcc, color='green', label='train accuracy')
    plt.plot(valAcc, color='blue', label='val accuracy')
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, 51, 5))
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()

    # plot loss trend
    plt.figure(figsize=(10, 7))
    plt.plot(trainLoss, color='orange', label='train loss')
    plt.plot(valLoss, color='red', label='val loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, 51, 5))
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


best_Loss = 10000.
trainAcc, trainLoss, valAcc, valLoss = train()
visulize(trainAcc, valAcc, trainLoss, valLoss)

#visualize_model(resnet, val_loader, 5)
