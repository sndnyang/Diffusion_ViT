
# Generative ViT and Hybrid ViT

pip install -r requirements.txt

The pretrained Hybrid ViT on CIFAR10, ImageNet 32x32, STL-10

https://drive.google.com/drive/folders/1QSkQaidk1tXZ_HDx8jEdnhQpBTSmckwC?usp=sharing 

## Training script


Please refer to scripts/cifar10_train.sh

```bash
python gevit_main.py --wd 0.05 \
      --heads 12 --depth 9 \
      --epochs 500 \
      --no_fid  \
      --dataset cifar10/cifar100/tinyimg/stl10/celeba/img32 \
      --data_path ./data  \
      --ps 4/8 \
      --gpu 0 \
      --px 100 --pyx 1
```
The patch size used in my experiments can be found at the bottom.

## Evaluation

### Accuracy

```shell
python eval_model.py --eval test_clf --ffnt 1 \
        --ps 4 \
        --dataset cifar10/cifar100/tinyimg/stl10/celeba/img32 \
        --data_path ./data  \
        --resume trained_models/cifar10_hybvit/ema_checkpoint.pth 
```

### Generate from scratch

It will compute the FID, so you still need to specify the data_path.

```shell
python eval_model.py --eval gen --ffnt 1 \
        --ps 4 \
        --dataset cifar10/cifar100/tinyimg/stl10/celeba/img32 \
        --data_path ./data  \
        --resume trained_models/cifar10_hybvit/ema_checkpoint.pth 
```


### Negative Log Likelihood 

nll or bits per dim (bpd)

```shell
python eval_model.py --eval nll --ffnt 1 --ps 4 --resume trained_models/cifar10_hybvit/ema_checkpoint.pth
```

### Calibration

ECE

```shell
python eval_model.py --eval cali --ffnt 1 --ps 4 --resume trained_models/cifar10_hybvit/ema_checkpoint.pth
```

### out-of-distribution detection

```shell
python eval_model.py --eval OOD --ood_dataset svhn --score_fn px --ffnt 1 --ps 4 --gpu-id 0 --resume $1 
```

### AUROC for OOD

```shell
python eval_model.py --eval logp_hist --datasets cifar10 svhn --ffnt 1 --ps 4 --resume $1 --gpu-id 0
```


### Attack

Please refer to scripts/bpda_attack.sh

```shell
CUDA_VISIBLE_DEVICES=0 python bpda_eot_attack.py  ckpt_path  l_inf/l_2  eps
```

# model config

| dataset   | params(Million) | patch size | dim        | heads | depth |
|-----------|-----------------|------------|------------|-------|-------|
| cifar10   | 11M             | 4 x 4      | 384        | 12    |   9   |
| cifar100  | 11M             | 4 x 4      | 384        | 12    |   9   |
| img32     | 11M             | 4 x 4      | 384        | 12    |   9   |
| tinyimg   | 11M             | 8 x 8      | 384        | 12    |   9   |
| stl10     | 13M             | 8 x 8      | 384        | 12    |   9   |
| celeba    | 17M             | 8 x 8      | 384        | 12    |   9   |
| img128-10 | 26M             | 8 x 8      | 512        | 12    |   9   |
| img224-10 | 84M             | 14 x 14    | 1024       | 12    |   9   |
