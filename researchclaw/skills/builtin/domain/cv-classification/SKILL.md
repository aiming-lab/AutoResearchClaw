---
name: cv-classification
description: Best practices for image classification tasks. Use when working on CIFAR, ImageNet, or other classification benchmarks.
metadata:
  category: domain
  trigger-keywords: "classification,image,cifar,imagenet,resnet,vision,cnn,vit"
  applicable-stages: "9,10"
  priority: "3"
  version: "1.0"
  author: researchclaw
  references: "He et al., Deep Residual Learning, CVPR 2016; Dosovitskiy et al., An Image is Worth 16x16 Words, ICLR 2021"
---

## Image Classification Best Practice
Architecture selection:
- Small scale (CIFAR-10/100): ResNet-18/34, WideResNet, Simple ViT
- Medium scale: ResNet-50, EfficientNet-B0/B1, DeiT-Small
- Large scale: ViT-B/16, ConvNeXt, Swin Transformer

Training recipe:
- Optimizer: AdamW (lr=1e-3 to 3e-4) or SGD (lr=0.1 with cosine decay)
- Weight decay: 0.01-0.1 for AdamW, 5e-4 for SGD
- Data augmentation: RandomCrop, RandomHorizontalFlip, Cutout/CutMix
- Warmup: 5-10 epochs linear warmup for transformers
- Batch size: 128-256 for CNNs, 512-1024 for ViTs (if memory allows)

Standard benchmarks:
- CIFAR-10: ~96% (ResNet-18), ~97% (WideResNet)
- CIFAR-100: ~80% (ResNet-18), ~84% (WideResNet)
- ImageNet: ~76% (ResNet-50), ~81% (ViT-B/16)
