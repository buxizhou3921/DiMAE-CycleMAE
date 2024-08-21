'''
将图片划分为训练集和测试集

训练集仅包含3个domain:painting、real、sketch
测试集包含6个domain,其中painting、real、sketch仅用于训练时的验证，clipart、infograph、quickdraw用于最终的性能评估

训练集文件组织结构
    - class1
    - class2
    - class3
    ...
测试集文件组织结构
    - domain1
        - class1
        - class2
        - class3
        ...
    - domain2
        ...
    - domain3
        ...
    ...
'''

import os
import shutil
from pathlib import Path


def main():
    src_data = './DomainNet/data'  # 原始数据路径
    class_names = ['airplane', 'bird', 'cooler', 'face', 'finger', 'fork', 'helmet', 'hot_dog', 'laptop', 'mouth',
                   'parrot', 'pizza', 'postcard', 'potato', 'rainbow', 'sink', 'star', 'sun', 'table',
                   'truck']  # 感兴趣的20个类别
    train_domain_names = ['painting', 'real', 'sketch']  # 训练、测试时使用到的3个domain
    test_domain_names = ['painting', 'real', 'sketch', 'clipart', 'infograph', 'quickdraw']  # 训练、测试时使用到的3个domain
    txt_dir = './DomainNet/split'
    train_dst_data = './DomainNet/trainset'  # 训练集路径
    test_dst_data = './DomainNet/testset'  # 测试集路径

    # 1.判断存储路径是否存在，若不存在，则创建；若存在，须为空，否则报错
    if Path(train_dst_data).exists():
        assert len(os.listdir(train_dst_data)) == 0, f'{train_dst_data} must be empty!'
    else:
        Path(train_dst_data).mkdir(parents=True)

    if Path(test_dst_data).exists():
        assert len(os.listdir(test_dst_data)) == 0, f'{test_dst_data} must be empty!'
    else:
        Path(test_dst_data).mkdir(parents=True)

    # 2.根据类别名创建子文件夹
    for cls_name in class_names:
        (Path(train_dst_data) / cls_name).mkdir()
    for d_name in test_domain_names:
        for cls_name in class_names:
            (Path(test_dst_data) / d_name / cls_name).mkdir(parents=True)

    # 3.读取txt文件进行数据集划分
    # trainset
    for d_name in train_domain_names:
        txt_file_name = Path(txt_dir) / (d_name + '_train.txt')
        with open(txt_file_name, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if Path((line.split(' ')[0])).parts[1] in class_names:
                src_img_name = Path(src_data) / (line.split(' ')[0])
                dst_img_name = Path(train_dst_data) / (line.split(' ')[0].split('/', 1)[1])
                shutil.copy(src_img_name, dst_img_name)

    # testset
    for d_name in test_domain_names:
        txt_file_name = Path(txt_dir) / (d_name + '_test.txt')
        with open(txt_file_name, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if Path((line.split(' ')[0])).parts[1] in class_names:
                src_img_name = Path(src_data) / (line.split(' ')[0])
                dst_img_name = Path(test_dst_data) / line.split(' ')[0]
                shutil.copy(src_img_name, dst_img_name)


if __name__ == '__main__':
    main()
