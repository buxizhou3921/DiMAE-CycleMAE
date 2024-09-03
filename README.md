# DiMAE-CycleMAE

### Data Preparation
- DomainNet
```
DiMAE-CycleMAE/DomainNet
├── data
│   ├── clipart
│   ├── infograph
│   └── painting
│   └── quickdraw
│   └── real
│   └── sketch
```

### Dataset Split
```
python data_split.py
```

### Requirements 
Pytorch == 1.8.0

Timm == 0.3.2

backbone: vit-large

### Pretrain 
```
python main_cyclemae.py
```

### Finetune
```
python main_finetune.py
```
