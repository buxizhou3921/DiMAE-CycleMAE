"""
CycleMAE预训练主函数
"""

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from domainnet import DomainNet
from cyclemae import CycleMAE
import util.lr_sched as lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def main():

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
    dataset_path = 'D:/LargeData/DomainNet/data'
    dataset = DomainNet(dataset_path)

    # 2.实例化DataLoader
    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  drop_last=True)

    # 3.实例化模型
    model = CycleMAE()
    model.cuda()

    loss_scaler = NativeScaler()

    # 4.实例化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    # 5.训练模型
    for epoch_idx in range(epoch_num):
        for iter_idx, batch_data in enumerate(train_dataloader):
            lr_sched.adjust_learning_rate(optimizer, iter_idx / len(train_dataloader) + epoch_idx, lr_cfg)

            mixed_data = torch.cat(batch_data[0]).cuda()
            original_data = torch.cat(batch_data[1]).cuda()
            label = []
            for item in batch_data[1]:
                label += list(item)
            loss = model(mixed_data, original_data)
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
            optimizer.zero_grad()
            print('loss: ', loss)


if __name__ == '__main__':
    main()
