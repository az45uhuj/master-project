import torch
import os
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import dataset
import copy

from sklearn.model_selection import train_test_split
from torchvision import models
from torch.utils.data import DataLoader

from tqdm import tqdm

#train_val, test = train_test_split(dataset.race_dataframe, test_size=0.2, random_state=42)
train, val = train_test_split(dataset.race_dataframe, test_size=0.2, random_state=42)

transform_objs = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

img_dir = 'F:\master thesis\\face dataset\Fairface\\' # face dataset folder
train_dataset = dataset.CustomImageDataset(train, img_dir, transform_objs)
val_dataset = dataset.CustomImageDataset(val,  img_dir, transform_objs)
#test_dataset = dataset.CustomImageDataset(test,  img_dir, transform_objs)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)
#test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

vgg16 = models.vgg16(pretrained=True)

# change the number of classes
vgg16.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))
vgg16.to(device)


# freeze convolution weights
for param in vgg16.features.parameters():
   param.requires_grad = True

# optimizer
optimizer = optim.SGD(vgg16.parameters(), lr=0.0001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# loss function
criterion = nn.CrossEntropyLoss()

# validation function
def validate(model, val_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for int, data in enumerate(val_dataloader):
        data, target = data[0].to(device), data[1].to(device)
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
        data, target = data[0].to(device), data[1].to(device)
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
    best_loss = 1000.

    start = time.time()
    for epoch in range(20):
        train_epoch_loss, train_epoch_accuracy = fit(vgg16, train_loader)

        val_epoch_loss, val_epoch_accuracy = validate(vgg16, val_loader)
        scheduler.step()

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(vgg16.state_dict(), 'E:\\genderModel\\asian/' +'asian_vgg16.pt') # save the best trained model

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
    plt.xticks(range(0, 21))
    plt.legend()
    #plt.savefig('accuracy.png')
    plt.show()

    # plot loss trend
    plt.figure(figsize=(10, 7))
    plt.plot(trainLoss, color='orange', label='train loss')
    plt.plot(valLoss, color='red', label='val loss')
    plt.xlabel('epoch')
    plt.xticks(range(0, 21))
    plt.legend()
    #plt.savefig('loss.png')
    plt.show()

### hellp
if __name__ == '__main__':
    best_Loss = 10000.
    trainAcc, trainLoss, valAcc, valLoss = train()
    visulize(trainAcc, valAcc, trainLoss, valLoss)
