"""
CycleMAE预训练主函数
"""

import torch
import os
import sys
import argparse
from torch.utils.data import DataLoader, random_split
from easydict import EasyDict
from domainnet import DomainNet
from cyclemae import CycleMAE
import util.lr_sched as lr_sched
from torch.utils.tensorboard import SummaryWriter
from util.misc import NativeScalerWithGradNormCount as NativeScaler


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--output_dir', default='./checkpoint', type=str, help='output_dir')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# lr = args.lr


def main():
    least_loss = sys.float_info.max

    # 0.一些超参数
    batch_size = 2
    num_workers = 1
    base_lr = 1e-3

    lr = base_lr * batch_size / 256
    epoch_num = 100
    warmup_epochs = 40
    min_lr = 0
    lr_cfg = EasyDict({'warmup_epochs': warmup_epochs, 'lr': lr, 'min_lr': min_lr})

    # 1.实例化Dataset
    dataset_path = '/home/rtx/sda1/kanghaidong/datasets/DomainNet/data'  # './DomainNet/data'
    dataset = DomainNet(dataset_path)

    # 1.1 划分训练集和测试集
    rate = 0.8  # 训练集占比
    train_size = int(rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 2.实例化DataLoader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True,
                                 drop_last=True)

    # 3.实例化模型
    model = CycleMAE()
    model.cuda()

    loss_scaler = NativeScaler()

    writer = SummaryWriter(log_dir='./logs')  # 这里的logs要与--logdir的参数一样

    # 4.实例化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    # 5.训练模型
    print('Train...')
    for epoch_idx in range(epoch_num):
        print("epoch_is:", epoch_idx)

        for iter_idx, batch_data in enumerate(train_dataloader):
            lr_sched.adjust_learning_rate(optimizer, iter_idx / len(train_dataloader) + epoch_idx, lr_cfg)

            mixed_data = torch.cat(batch_data[0]).cuda()
            original_data = torch.cat(batch_data[1]).cuda()
            # label = []
            # for item in batch_data[1]:
            #     label += list(item)
            loss = model(mixed_data, original_data)
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
            optimizer.zero_grad()
            print('loss: ', loss)
            # writer.add_scalar('data/loss', loss, iter_idx)
        writer.add_scalar('data/train_loss', loss, epoch_idx)

        # Eval...
        with torch.no_grad():
            for iter_idx, batch_data in enumerate(train_dataloader):
                mixed_data = torch.cat(batch_data[0]).cuda()
                original_data = torch.cat(batch_data[1]).cuda()
                loss = model(mixed_data, original_data)

        writer.add_scalar("data/test_loss", loss, epoch_idx)

        # Save checkpoint.
        if loss < least_loss:
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            least_loss = loss
            state = {
                'model': model.state_dict(),
                'loss': loss,
                'epoch': epoch_idx,
                'optimizer': optimizer.state_dict(),
            }
            print('Saving..')
            torch.save(state, './checkpoint/ckpt.pth')

    writer.close()  # 执行close立即刷新，否则将每120秒自动刷新


if __name__ == '__main__':
    main()
    print("--------Finish--------")
