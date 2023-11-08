"""
DomainNet数据读取
"""
import os
from pathlib import Path
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DomainNet(Dataset):
    CLASSES = ['airplane', 'bird', 'cooler', 'face', 'finger',
               'fork', 'helmet', 'hot_dog', 'laptop', 'mouth',
               'parrot', 'pizza', 'postcard', 'potato', 'rainbow',
               'sink', 'star', 'sun', 'table', 'truck']  # 选择20个类别，这里先任意选，后面根据论文实验细节再修改
    DOMAIN = ['real', 'painting', 'quickdraw']

    def __init__(self, dataset_path):
        self.data_path = dataset_path
        image_num_domains = [0 for _ in range(len(self.DOMAIN))]

        # 获取domain中图片数量最多的domain，让该图片数作为Dataset长度
        for domain_idx, domain in enumerate(self.DOMAIN):
            for cls_name in self.CLASSES:
                image_num_domains[domain_idx] += len(os.listdir(Path(self.data_path) / domain / cls_name))
        self.length = max(image_num_domains)
        # max_length_domain = self.DOMAIN[torch.argmax(torch.tensor(image_num_domains)).item()]

        # 获取每个domain中，每个cls的第一张图在img_name列表中的位置
        self.cls_loc_domain_dict = {}
        for domain in self.DOMAIN:
            self.cls_loc_domain_dict[domain] = [0]
            for cls_name in self.CLASSES[:-1]:
                cls_loc = len(os.listdir(Path(self.data_path) / domain / cls_name)) + self.cls_loc_domain_dict[domain][
                    -1]
                self.cls_loc_domain_dict[domain].append(cls_loc)

        # 获取每个domain的图片名列表
        self.domain_image_dict = {}
        for domain in self.DOMAIN:
            self.domain_image_dict[domain] = []
            for cls_name in self.CLASSES:
                self.domain_image_dict[domain] += sorted(os.listdir(Path(self.data_path) / domain / cls_name))

        # 获取每个domain中，每个cls的图片名列表
        self.domain_image_per_cls_dict = {}
        for domain in self.DOMAIN:
            self.domain_image_per_cls_dict[domain] = {}
            for cls_name in self.CLASSES:
                self.domain_image_per_cls_dict[domain][cls_name] = sorted(
                    os.listdir(Path(self.data_path) / domain / cls_name))

        # max_length_domain_img_name_list = []
        # for cls_name in self.CLASSES:
        #     max_length_domain_img_name_list += sorted(os.listdir(Path(self.data_path) / max_length_domain / cls_name))

    def __getitem__(self, index):
        image_list = []
        label_list = []
        aux_lambda = 0.3

        for main_domain in self.DOMAIN:
            main_img_name = self.domain_image_dict[main_domain][index % len(self.domain_image_dict[main_domain])]
            main_img_label = self.CLASSES[(index % len(self.domain_image_dict[main_domain]) - torch.tensor(
                self.cls_loc_domain_dict[main_domain]) >= 0).sum().item() - 1]
            main_img_name = Path(self.data_path) / main_domain / main_img_label / main_img_name

            aux_img_name_list = []
            for aux_domain in self.DOMAIN:  # 获取要混合的domain中与main_img_name同等类别的图片，随机获取
                if aux_domain != main_domain:
                    aux_img_name_list.append(Path(self.data_path) / aux_domain / main_img_label / (
                        random.sample(self.domain_image_per_cls_dict[aux_domain][main_img_label], 1)[0]))

            # Content Preserved Style-Mix
            # 1.读取main img
            main_img = cv2.imread(str(main_img_name))
            main_img_resize = cv2.resize(main_img, (224, 224))
            main_img_fre = np.fft.fft2(main_img_resize, axes=(0, 1))
            main_fre_m = np.abs(main_img_fre)  # 幅度谱，求模得到
            main_fre_p = np.angle(main_img_fre)  # 相位谱，求相角得到
            img_after_aux_list = []
            for aux_img_name in aux_img_name_list:
                # 2.读aux img
                aux_img = cv2.imread(str(aux_img_name))
                aux_img_resize = cv2.resize(aux_img, (224, 224))
                aux_img_fre = np.fft.fft2(aux_img_resize, axes=(0, 1))
                aux_fre_m = np.abs(aux_img_fre)  # 要混合的图像的幅度
                # 3.幅度混合
                A = aux_lambda * aux_fre_m + (1 - aux_lambda) * main_fre_m  # 混合后的幅度
                # 4.得到混合后的图，暂存在列表中
                fre_after_aux = A * np.e ** (1j * main_fre_p)  # 把幅度谱和相位谱再合并为复数形式的频域图数据
                img_after_aux = np.abs(np.fft.ifft2(fre_after_aux, axes=(0, 1)))  # 还原为空间域图像
                img_after_aux_list.append(img_after_aux)

            # 5.mixup
            mixup_weight = 1.0 / len(img_after_aux_list)
            mixup_img = np.zeros((224, 224, 3), dtype=np.float64)
            for img_after_aux in img_after_aux_list:
                mixup_img = mixup_img + mixup_weight * img_after_aux

            mixup_img = mixup_img.astype(np.uint8)
            image_list.append(mixup_img)
            label_list.append(main_img_label)

        image_tensor_list = []

        for image in image_list:
            # BGR转RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 转为Tensor
            image_tensor = to_tensor(image_rgb)
            # 减均值、除以标准差
            image_tensor = normalize(image_tensor, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            image_tensor_list.append(image_tensor)

        return image_tensor_list, label_list
        # return image_set, label_set

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = DomainNet('D:LargeData/DomainNet/data')

    # dataset[8888]
    print('-----')