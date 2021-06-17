import copy

import torch
import tqdm
from torch.utils.data import DataLoader

from utils.dataset import VOCDataset
from nets.darknet import DarkNet
from nets.yolo import YOLOv1
from utils.loss import Loss

import os
import numpy as np
import math

# Check if GPU devices are available.
print(f'CUDA DEVICE COUNT: {torch.cuda.device_count()}')

# Path to data dir.
base_dir = '../VOCDataset'

# Path to save weights
log_dir = 'weights'
os.makedirs(log_dir, exist_ok=True)

# Path to label files.
train_files = 'train.txt'
test_files = 'test.txt'

# Frequency to print/log the results.
print_freq = 5
tb_log_freq = 5

# Training hyper parameters.
init_lr = 0.001
base_lr = 0.01
# momentum = 0.9
# weight_decay = 5.0e-4
num_epochs = 100
batch_size = 64
seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Learning rate scheduling.
def update_lr(optimizer, epoch, burning_base, burning_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burning_base, burning_exp)
    elif epoch == 1:
        lr = base_lr
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
darknet.features = torch.nn.DataParallel(darknet.features)

# Load YOLO model.
net = YOLOv1(darknet.features).to(device)
net.conv_layers = torch.nn.DataParallel(net.conv_layers)

# Setup loss and optimizer.
criterion = Loss(feature_size=net.feature_size)
optimizer = torch.optim.SGD(net.parameters(), lr=init_lr)  # , momentum=momentum, weight_decay=weight_decay)

# Load Pascal-VOC dataset.
with open(f'{base_dir}/{train_files}') as f:
    train_names = f.readlines()
train_dataset = VOCDataset(True, file_names=train_names, base_dir=base_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

with open(f'{base_dir}/{test_files}') as f:
    test_names = f.readlines()
test_dataset = VOCDataset(False, file_names=test_names, base_dir=base_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print('Number of training images: ', len(train_dataset))

# Training loop.
best_val_loss = np.inf

for epoch in range(num_epochs):
    print('\n')
    print('Starting epoch {} / {}'.format(epoch, num_epochs))

    # Training.
    net.train()
    total_loss = 0.0
    total_batch = 0
    print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
    progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, targets) in progress_bar:
        # Update learning rate.
        update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1))
        lr = get_lr(optimizer)

        # Load data as a batch.
        batch_size_this_iter = images.size(0)
        images, targets = images.to(device), targets.to(device)

        # Forward to compute loss.
        predictions = net(images)
        loss = criterion(predictions, targets)
        loss_this_iter = loss.item()
        total_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter

        # Backward to update model weight.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # Print current loss.
        # if i % print_freq == 0:
        #     print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
        #           % (epoch, num_epochs, i, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))

        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, num_epochs), total_loss / (i + 1), mem)
        progress_bar.set_description(s)

    # Validation.
    net.eval()
    val_loss = 0.0
    total_batch = 0

    progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (images, targets) in progress_bar:
        # Load data as a batch.
        batch_size_this_iter = images.size(0)
        images, targets = images.to(device), targets.to(device)

        # Forward to compute validation loss.
        with torch.no_grad():
            predictions = net(images)
        loss = criterion(predictions, targets)
        loss_this_iter = loss.item()
        val_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter
    val_loss /= float(total_batch)

    # Save results.
    save = {'state_dict': net.state_dict()}
    torch.save(save, os.path.join(log_dir, 'final.pth'))
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        save = {'state_dict': net.state_dict()}
        torch.save(save, os.path.join(log_dir, 'best.pth'))

    # Print.
    print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f'
          % (epoch + 1, num_epochs, val_loss, best_val_loss))
