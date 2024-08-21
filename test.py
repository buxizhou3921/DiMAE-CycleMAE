from loguru import logger
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomGrayscale, RandomErasing, \
    Normalize
from classifier import Classifier


def main():
    # 0.参数
    num_classes = 20
    weight_file = './cls_model_1000/cls_model_epoch_5.pth'
    test_dataset_root_clipart = './DomainNet/testset/clipart'
    test_dataset_root_infograph = './DomainNet/testset/infograph'
    test_dataset_root_quickdraw = './DomainNet/testset/quickdraw'
    num_workers = 4
    batch_size = 16

    # 1.加载权重
    ckpt = torch.load(weight_file)

    # 2.实例化分类模型
    cls_model = Classifier(num_classes=num_classes)
    cls_model.eval()
    cls_model.cuda()
    cls_model.load_state_dict(ckpt)

    # 3.构造Dataset和DataLoader
    dataset_test_clipart = ImageFolder(test_dataset_root_clipart,
                                       transform=Compose([ToTensor(),
                                                          Resize((224, 224)),
                                                          Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])]))

    dataset_test_infograph = ImageFolder(test_dataset_root_infograph,
                                         transform=Compose([ToTensor(),
                                                            Resize((224, 224)),
                                                            Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])]))

    dataset_test_quickdraw = ImageFolder(test_dataset_root_quickdraw,
                                         transform=Compose([ToTensor(),
                                                            Resize((224, 224)),
                                                            Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])]))

    test_dataloader_clipart = DataLoader(dataset=dataset_test_clipart,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=False)

    test_dataloader_infograph = DataLoader(dataset=dataset_test_infograph,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           drop_last=False)

    test_dataloader_quickdraw = DataLoader(dataset=dataset_test_quickdraw,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           drop_last=False)

    # 4.测试
    val_result_dict = test(cls_model, test_dataloader_clipart, num_classes, batch_size)
    print('clipart eval acc:', val_result_dict)
    val_result_dict = test(cls_model, test_dataloader_infograph, num_classes, batch_size)
    print('infograph eval acc:', val_result_dict)
    val_result_dict = test(cls_model, test_dataloader_quickdraw, num_classes, batch_size)
    print('quickdraw eval acc:', val_result_dict)


def test(model, val_dataloader, num_classes, batch_size):
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

    logger.info(f'mean_acc: {round(correct_num / total_img, 4)}')
    logger.info(f'class_acc: {[round(v.item(), 4) for v in acc_per_class]}')

    model.train()

    return {'mean_acc': round(correct_num / total_img, 4),
            'mean_loss': total_loss / (len(val_dataloader.dataset) // batch_size)}


if __name__ == '__main__':
    main()
