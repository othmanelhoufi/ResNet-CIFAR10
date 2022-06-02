#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from torchinfo import summary
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from resnet_architecture import ResNet18

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb param
WANDB = 1
ENTITY = 'othmanelhoufi'
EXPERIMENT_NAME = 'ResNet-18-with-MultiStepLR'

# Hyper-parameters
batch_size = 100
num_epochs = 200
learning_rate = 0.1

# fetch and split CIFAR10 dataset
def init_dataset_splits(batch_size):
    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])


    # CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

    torch.manual_seed(43)
    val_size = 5000
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def show_sample_images(train_loader):
    for images, _ in train_loader:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
        # plt.show()
        break


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model,criterion, val_loader):
    val_steps = []
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        val_steps.append({'val_loss': loss.detach(), 'val_acc': acc})

    batch_losses = [x['val_loss'] for x in val_steps]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in val_steps]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def training_loop(model, optimizer, criterion, learning_rate, scheduler, train_loader, val_loader):

    # For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Magic
    if WANDB: wandb.watch(model, log_freq=100)

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    model.train()
    for epoch in tqdm(range(num_epochs)):
        # log lr
        if WANDB: wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

        # losses history
        losses = []

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}" .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # decaying lr
        scheduler.step()

        if WANDB: wandb.log({"train_loss": torch.stack(losses).mean()})

        # Decay learning rate
        # if (epoch+1) % 20 == 0:
        #     curr_lr /= 3
        #     update_lr(optimizer, curr_lr)

        # Validation phase
        eval_result = evaluate(model, criterion, val_loader)
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch+1, eval_result['val_loss'], eval_result['val_acc']))
        if WANDB:
            wandb.log(eval_result)
            wandb.log({"epoch": epoch})



def log_confusion_matrix(y_pred, y_true):
    # constant for classes
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    if WANDB: wandb.log({"Confusion Matrix : " + EXPERIMENT_NAME: wandb.Image(plt.gcf())})

def testing_loop(model, test_loader):
    # constructing confusion matrix
    y_pred = []
    y_true = []

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy() )

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    log_confusion_matrix(y_pred, y_true)


def main():
    model = ResNet18().to(device)
    train_loader, val_loader, test_loader = init_dataset_splits(batch_size)
    show_sample_images(train_loader)

    summary(model, input_size=(1, 3, 32, 32))

    # Loss and Opt
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,150], gamma=0.1)

    training_loop(model, optimizer, criterion, learning_rate, scheduler, train_loader, val_loader)
    testing_loop(model, test_loader)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'resnet.ckpt')

if __name__ == '__main__':

    if WANDB:
        # init wandb
        wandb.init(project='ResNet-Architecture', name=EXPERIMENT_NAME, entity=ENTITY)
    main()
