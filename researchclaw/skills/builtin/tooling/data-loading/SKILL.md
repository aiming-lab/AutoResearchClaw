---
name: data-loading
description: Optimize data loading pipeline to prevent GPU starvation. Use when setting up DataLoader or data preprocessing.
metadata:
  category: tooling
  trigger-keywords: "data,loading,dataloader,dataset,preprocessing,augmentation"
  applicable-stages: "10"
  priority: "6"
  version: "1.0"
  author: researchclaw
  references: "PyTorch Data Loading Tutorial, pytorch.org"
---

## Efficient Data Loading Best Practice
1. Use num_workers = min(8, os.cpu_count()) for DataLoader
2. Enable pin_memory=True when using GPU
3. Use persistent_workers=True to avoid re-spawning
4. Pre-compute and cache transformations when possible
5. For image data: use torchvision.transforms.v2 (faster)
6. For large datasets: consider memory-mapped files or WebDataset
7. Profile with torch.utils.bottleneck to find I/O bottlenecks
