import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/RAF/')
import dataset

from sklearn.model_selection import train_test_split
from torchvision import models
from torch.utils.data import DataLoader


#train_val, test = train_test_split(datasets.landmarks_dataframe, test_size=0.2, random_state=42)
train, val = train_test_split(dataset.landmarks_dataframe, test_size=0.2, random_state=42)
#dataset.RandomVerticalFlip(),dataset.RandomHorizontalFlip(),
transform_objs = transforms.Compose([dataset.ToPILImage(), dataset.Rescale(224),dataset.ToTensor()])
#transform_objs = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
#transform_objs2 = transforms.Compose([datasets.Rescale(224)])

img_dir = 'F:\master thesis\\face dataset\RAF\\basic\Image\original\\'
train_dataset = dataset.CustomImageDataset(train, img_dir, transform_objs)
val_dataset = dataset.CustomImageDataset(val,  img_dir, transform_objs)
#test_dataset = datasets.CustomImageDataset(test,  img_dir, transform_objs)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)
#test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

resnet = models.resnet34(pretrained=True)

# freeze convolution weights
for param in resnet.parameters():
   param.requires_grad = True

num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 10) # 5 landmarks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# optimizer
optimizer = optim.SGD(resnet.parameters(), lr=0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# loss function
criterion = nn.MSELoss()

# validation function
def validate(model, val_dataloader):
    model.eval()
    val_running_loss = 0.0
    for int, data in enumerate(val_dataloader):
        data, target = data['image'].float().to(device), data['landmarks'].float().to(device)
        output = model(data)
        output = output.reshape(-1, 5, 2)
        loss = criterion(output, target)
        val_running_loss += loss.item()
        #_, preds = torch.max(output.data, 1)
    val_loss = val_running_loss / len(val_dataloader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')
    return val_loss

# training function
def fit(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        data, target = data['image'].float().to(device), data['landmarks'].float().to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.reshape(-1, 5, 2)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        #_, preds = torch.max(output.data, 1)
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(train_dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}')
    return train_loss

def train():
    train_loss = []
    val_loss = []
    best_loss = 1000
    #best_model_wts = copy.deepcopy(vgg16.state_dict())
    start = time.time()
    for epoch in range(20):
        train_epoch_loss = fit(resnet, train_loader)
        #scheduler.step()
        val_epoch_loss = validate(resnet, val_loader)
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(resnet.state_dict(), 'E:\\RAF_landmarks\\' +'mix_resnet34.pt')
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
    end = time.time()
    print((end - start) / 60, 'minutes')
    return train_loss, val_loss

def visulize(trainLoss, valLoss):
    # plot loss trend
    plt.figure(figsize=(10, 7))
    plt.plot(trainLoss, color='orange', label='train loss')
    plt.plot(valLoss, color='red', label='val loss')
    plt.xticks(range(0, 21))
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('E:\\RAF_landmarks\\mix_loss.png')
    plt.show()

def show_landmarks_2(image, landmarks_gt, landmarks_pred, ax_new):
    """Show image with landmarks"""
    ax_new.imshow(image)
    ax_new.scatter(landmarks_gt[:, 0], landmarks_gt[:, 1], s=20, marker='.', c=['g'])
    ax_new.scatter(landmarks_pred[:, 0], landmarks_pred[:, 1], s=20, marker='.', c=['r'])

def visualize_model(model, dataloader, num_images=1):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
      for sample in dataloader:
        inputs = sample['image'].float().to(device)
        labels = sample['landmarks'].float().to(device)

        outputs = model(inputs)
        preds = outputs
        for j in range(inputs.size()[0]):
            lms = preds.cpu().data[j].reshape(-1, 2)
            lms_gt = labels.cpu().data[j].reshape(-1, 2)
            images_so_far += 1
            img = inputs.cpu().data[j].permute(1, 2, 0)
            ax = plt.subplot(1, num_images, images_so_far)
            ax.imshow(img)
            ax.set_title('Loss {:.3f}'.format(((lms_gt-lms)**2).mean().item()))
            ax.axis('off')
            show_landmarks_2(np.array(img), np.array(lms_gt), np.array(lms), ax)

            plt.tight_layout()
            if images_so_far == num_images:
                model.train(mode=was_training)
                plt.savefig('E:\\RAF_landmarks\\mix_example.png')
                plt.show()
                return
        model.train(mode=was_training)
        plt.show()

if __name== '__main__':
    best_Loss = 10000.
    trainLoss, valLoss = train()
    visulize(trainLoss, valLoss)
    visualize_model(resnet, val_loader, 5)