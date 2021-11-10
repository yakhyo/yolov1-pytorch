import torch
from torch.utils.data import DataLoader

from utils.dataset import VOCDataset
from nets.nn import YOLOv1
from utils.loss import Loss

import os
import math
import tqdm
import argparse
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='YOLOv1 implementation using PyTorch')
parser.add_argument('--base_dir', default='../../Datasets/VOC/', required=False, help='Path to data dir')
parser.add_argument('--log_dir', default='./weights', required=False, help='Path to save weights')
parser.add_argument('--init_lr', default=0.001, required=False, help='Initial learning rate')
parser.add_argument('--base_lr', default=0.01, required=False, help='Base learning rate')
parser.add_argument('--momentum', default=0.9, required=False, help='Momentum')
parser.add_argument('--weight_decay', default=5.0e-4, required=False, help='Weight decay')
parser.add_argument('--num_epochs', default=135, required=False, help='Number of epochs')
parser.add_argument('--batch_size', default=64, required=False, help='Batch size')
parser.add_argument('--seed', default=42, required=False, help='Random seed')

args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Check if GPU devices are available.
print(f'CUDA DEVICE COUNT: {torch.cuda.device_count()}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Learning rate scheduling.
def update_lr(optimizer, epoch, burning_base, burning_exp=4.0):
    if epoch == 0:
        lr = args.init_lr + (args.base_lr - args.init_lr) * math.pow(burning_base, burning_exp)
    elif epoch == 1:
        lr = args.base_lr
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


def train():
    # Load YOLO model.
    net = YOLOv1().to(device)
    net = torch.nn.DataParallel(net)

    accumulate = max(round(64 / args.batch_size), 1)

    params = defaultdict()
    params['weight_decay'] = args.weight_decay
    params['weight_decay'] *= args.batch_size * accumulate / 64

    pg0, pg1, pg2 = [], [], []
    for k, v in net.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            pg1.append(v.weight)

    optimizer = torch.optim.SGD(pg0, lr=args.init_lr, momentum=args.momentum, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': pg2})

    # Setup loss and optimizer.
    criterion = Loss()

    # Load Pascal-VOC dataset.
    with open(f'{args.base_dir}/train.txt') as f:
        train_names = f.readlines()
    train_dataset = VOCDataset(True, file_names=train_names, base_dir=args.base_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    with open(f'{args.base_dir}/test.txt') as f:
        test_names = f.readlines()
    test_dataset = VOCDataset(False, file_names=test_names, base_dir=args.base_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size // 2, shuffle=False, num_workers=4)

    print('Number of training images: ', len(train_dataset))

    # Training loop.
    best_val_loss = np.inf

    for epoch in range(args.num_epochs):
        print('\n')
        print('Starting epoch {} / {}'.format(epoch, args.num_epochs))

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

            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, args.num_epochs), total_loss / (i + 1), mem)
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
        torch.save(save, os.path.join(args.log_dir, 'final.pth'))
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            save = {'state_dict': net.state_dict()}
            torch.save(save, os.path.join(args.log_dir, 'best.pth'))

        # Print.
        print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f'
              % (epoch + 1, args.num_epochs, val_loss, best_val_loss))


if __name__ == '__main__':
    train()
