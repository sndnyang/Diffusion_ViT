
# Your ViT is Secretly a Hybrid Discriminative-Generative Diffusion Model

PyTorch implementation of "Your ViT is Secretly a Hybrid Discriminative-Generative Diffusion Model" https://arxiv.org/abs/2208.07791

It contains GenViT(Generative ViT) and HybViT (Hybrid ViT)

## Configuration

pip install -r requirements.txt

The pretrained Hybrid ViT on CIFAR10, ImageNet 32x32, STL-10

https://drive.google.com/drive/folders/1QSkQaidk1tXZ_HDx8jEdnhQpBTSmckwC?usp=sharing 

I find a new paper [U-ViT](https://arxiv.org/abs/2209.12152) achieves a FID 3.11 on CIFAR10, which is significantly better than 20.20 in my work.


## Training Script

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
The default patch sizes used in experiments can be found at the bottom.

## Evaluation

### Accuracy

```shell
python eval_model.py --eval test_clf --ffnt 1 \
        --ps 4 \
        --dataset cifar10/cifar100/tinyimg/stl10/celeba/img32 \
        --data_path ./data  \
        --resume trained_models/cifar10_hybvit/ema_checkpoint.pth 
```

### Generate From Scratch

It will compute the FID, so you still need to specify the data_path. I didn't try any fast sampling methods.

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

### Out-of-Distribution Detection

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

# Model Config

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

**Note**

[U-ViT](https://arxiv.org/abs/2209.12152) easily outperforms this work by a large margin, and close the gap to UNet-based DDPM.

They use a vanilla ViT to achieve a FID 5.97, which is significantly better than 20.20 in my work. I think it's because my code/coding is much weaker, not the model capacity/patch size. 

It's interesting to see more promising work on high-resolution datasets.

# Citation

If you found this work useful and used it on your own research, please consider citing this paper.

```
@misc{yang2022vit,
      title={Your ViT is Secretly a Hybrid Discriminative-Generative Diffusion Model}, 
      author={Xiulong Yang and Sheng-Min Shih and Yinlin Fu and Xiaoting Zhao and Shihao Ji},
      year={2022},
      eprint={2208.07791},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


# Knowledgement

The code is built upon 

1. [SL_ViT](https://github.com/aanna0701/SPT_LSA_ViT) for vanilla ViT backbone
2. [PyTorch Diffision Framework](https://github.com/lucidrains/denoising-diffusion-pytorch)
3. NLL(Negative Log Likelihood) bits per dim(bits/dim) [guidance diffusion](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py)
