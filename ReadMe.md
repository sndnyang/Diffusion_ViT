
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

Configurations to reproduce (some small changes)

python eval_gevit.py --resume pretrained_models/cifar10_hybrid/ema_checkpoint.pth --ffnt


python eval_gevit.py --resume pretrained_models/celeba_gen/ema_checkpoint.pth --ffnt --dataset celeba