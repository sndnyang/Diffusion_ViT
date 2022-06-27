
# Evaluate Generative ViT

pretrained models can be downloaded from:

https://nitro2.cs.gsu.edu/dddown/

The list of pretrained models (gen: trained as generator only, hybrid: generator and classifier) :

- https://nitro2.cs.gsu.edu/dddown/celeba_gen/ema_checkpoint.pth
- https://nitro2.cs.gsu.edu/dddown/cifar10_gen/ema_checkpoint.pth
- https://nitro2.cs.gsu.edu/dddown/cifar10_hybrid/ema_checkpoint.pth
- https://nitro2.cs.gsu.edu/dddown/imagenet10_gen/ema_checkpoint.pth
- https://nitro2.cs.gsu.edu/dddown/img12810_gen/ema_checkpoint.pth
- https://nitro2.cs.gsu.edu/dddown/stl10_hybrid/ema_checkpoint.pth
- https://nitro2.cs.gsu.edu/dddown/stl10_gen/ema_checkpoint.pth


Scripts/Configurations to reproduce (models may have some small changes)

```
python eval_gevit.py --resume pretrained_models/cifar10_hybrid/ema_checkpoint.pth 
```

```
python eval_gevit.py --resume pretrained_models/celeba_gen/ema_checkpoint.pth --dataset celeba
```


if don't want to download STL for classification
```
python eval_gevit.py --resume pretrained_models/stl10_hybrid/ema_checkpoint.pth --dataset stl10 --gpu-id 4 --data_path none
```


I add time embeddings to features before FFN layer(doesn't help, but need --ffnt)
```
python eval_gevit.py --resume pretrained_models/imagenet10_gen/ema_checkpoint.pth --dataset img10 --gpu-id 4 --data_path no --dim 1024  --ps 14 --ffnt
```

# model config

| dataset   | params(Million) | patch size | dim        | heads | depth |
|-----------|-----------------|------------|------------|-------|-------|
| cifar10   | 11M             | 4 x 4      | 384        |       |       |
| celeba    | 17M             | 8 x 8      | 384        |       |       |
| stl10     | 13M             | 8 x 8      | 384        |       |       |
| img128-10 | 26M             | 8 x 8      | 512        |       |       |
| img224-10 | 84M             | 14 x 14    | 1024       |       |       |
