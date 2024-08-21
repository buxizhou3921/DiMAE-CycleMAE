from pathlib import Path
from collections import OrderedDict
from easydict import EasyDict
from loguru import logger
from tqdm import tqdm

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomGrayscale, RandomErasing, \
    Normalize
from torch.utils.tensorboard import SummaryWriter

from classifier import Classifier
import util.lr_sched as lr_sched


def main():
    # 0.参数

    train_dataset_root = './DomainNet/trainset'
    val_dataset_root_sketch = './DomainNet/testset/sketch'
    val_dataset_root_real = './DomainNet/testset/real'
    val_dataset_root_painting = './DomainNet/testset/painting'
    run_output = './cls_model_1000'

    batch_size = 16
    warmup_epochs = 10
    num_epoch = 100
    num_workers = 4
    min_lr = 0.0001
    lr = 0.001
    lr_cfg = EasyDict({'warmup_epochs': warmup_epochs, 'lr': lr, 'min_lr': min_lr, 'epochs': num_epoch})
    num_classes = 20
    writer = SummaryWriter(log_dir=run_output)

    # 1.读取预训练权重
    weight_file = './checkpoint/ckpt1000.pth'
    ckpt = torch.load(weight_file)

    # 2.实例化分类模型
    cls_model = Classifier(num_classes=num_classes)
    cls_model.cuda()

    # 3.加载预训练权重
    encoder_state_dict = OrderedDict()

    for k, v in ckpt['model'].items():
        if k.split('.', 1)[0] == 'encoder':
            encoder_state_dict[k] = v
    cls_model.load_state_dict(encoder_state_dict, strict=False)

    # 4.构建DataLoader

    dataset_train = ImageFolder(train_dataset_root,
                                transform=Compose([
                                    RandomHorizontalFlip(p=0.3),
                                    RandomGrayscale(p=0.3),
                                    ToTensor(),
                                    Resize((224, 224)),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    RandomErasing(p=0.1, scale=(0.02, 0.15))
                                ]))

    train_dataloader = DataLoader(dataset=dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    dataset_val_sketch = ImageFolder(val_dataset_root_sketch,
                                     transform=Compose([ToTensor(),
                                                        Resize((224, 224)),
                                                        Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])]))

    dataset_val_real = ImageFolder(val_dataset_root_real,
                                   transform=Compose([ToTensor(),
                                                      Resize((224, 224)),
                                                      Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]))

    dataset_val_painting = ImageFolder(val_dataset_root_painting,
                                       transform=Compose([ToTensor(),
                                                          Resize((224, 224)),
                                                          Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])]))

    val_dataloader_sketch = DataLoader(dataset=dataset_val_sketch,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       drop_last=False)

    val_dataloader_real = DataLoader(dataset=dataset_val_real,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=False)

    val_dataloader_painting = DataLoader(dataset=dataset_val_painting,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=False)

    # 5.实例化优化器
    optimizer = torch.optim.AdamW(cls_model.parameters(), lr=lr)

    # 6.实例化损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 7.训练
    if Path(run_output).exists() is False:
        Path(run_output).mkdir(parents=True)

    total_epoch = num_epoch
    best_val_mean_acc = 0
    for current_epoch in range(total_epoch):
        total_loss = 0
        pbar = tqdm(total=len(train_dataloader), desc=f'epoch {current_epoch}')
        for iter_idx, batch_data in enumerate(train_dataloader):
            pbar.update(1)
            lr_sched.adjust_learning_rate(optimizer, iter_idx / len(train_dataloader) + current_epoch, lr_cfg)
            optimizer.zero_grad()

            img = batch_data[0].cuda()
            label = batch_data[1].cuda()

            out = cls_model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss, iter_idx + current_epoch * len(train_dataloader))

        pbar.close()

        val_result_dict_sketch = validate(cls_model, val_dataloader_sketch, num_classes, current_epoch, batch_size,
                                          writer, 'sketch eval acc')
        # print('sketch eval acc:', val_result_dict_sketch)
        val_result_dict_real = validate(cls_model, val_dataloader_real, num_classes, current_epoch, batch_size, writer,
                                        'real eval acc')
        # print('real eval acc:', val_result_dict_real)
        val_result_dict_painting = validate(cls_model, val_dataloader_painting, num_classes, current_epoch, batch_size,
                                            writer, 'painting eval acc')
        # print('painting eval acc:', val_result_dict_painting)

        val_mean_acc = (val_result_dict_sketch['mean_acc'] + val_result_dict_real['mean_acc'] +
                        val_result_dict_painting['mean_acc']) / 3
        if val_mean_acc > best_val_mean_acc:
            best_val_mean_acc = val_mean_acc
            save_model(cls_model, f'{run_output}/cls_model_best.pth')

        save_model(cls_model, f'{run_output}/cls_model_epoch_{current_epoch}.pth')
    save_model(cls_model, f'{run_output}/cls_model_final.pth')


def validate(model, val_dataloader, num_classes, current_epoch, batch_size, tb_writer, tb_writer_name):
    model.eval()
    total_img = len(val_dataloader.dataset)
    gt_label = []
    pred_label = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    total_loss = 0
    for batch_data in val_dataloader:
        img = batch_data[0].cuda()
        model_out = model(img)
        out = softmax(model_out, dim=-1)
        preds = out.max(dim=-1)[1]
        gt_label.extend(batch_data[1].tolist())
        pred_label.extend(preds.tolist())
        total_loss = total_loss + cross_entropy_loss(model_out, batch_data[1].cuda()).item()

    correct_num = (torch.tensor(gt_label) == torch.tensor(pred_label)).sum().item()
    incorrect_num = (torch.tensor(gt_label) != torch.tensor(pred_label)).sum().item()
    acc_per_class = []

    for i in range(num_classes):
        acc_per_class.append(((torch.tensor(gt_label) == torch.tensor(pred_label))[torch.tensor(gt_label) == i]).sum() /
                             (torch.tensor(gt_label) == i).sum().item())

    # logger.info(f'mean_acc: {round(correct_num/total_img, 4)}')
    # logger.info(f'class_acc: {[round(v.item(), 4) for v in acc_per_class]}')

    model.train()
    tb_writer.add_scalar(tb_writer_name, round(correct_num / total_img, 4), current_epoch)

    return {'mean_acc': round(correct_num / total_img, 4),
            'mean_loss': total_loss / (len(val_dataloader.dataset) // batch_size)}


def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


if __name__ == '__main__':
    main()