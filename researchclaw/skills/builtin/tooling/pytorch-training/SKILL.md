---
name: pytorch-training
description: Best practices for building robust PyTorch training loops. Use when generating or reviewing ML training code.
metadata:
  category: tooling
  trigger-keywords: "training,pytorch,torch,deep learning,neural network,model"
  applicable-stages: "10,12"
  priority: "3"
  version: "1.0"
  author: researchclaw
  references: "PyTorch Performance Tuning Guide, pytorch.org"
  code-template: |
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch['input']), batch['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
---

## PyTorch Training Best Practice
1. Use torch.manual_seed() for reproducibility (set for torch, numpy, random)
2. Use DataLoader with num_workers>0 and pin_memory=True for GPU
3. Enable cudnn.benchmark=True for fixed input sizes
4. Use learning rate schedulers (CosineAnnealingLR or OneCycleLR)
5. Implement early stopping based on validation metric
6. Log metrics every epoch, save best model checkpoint
7. Use torch.no_grad() for evaluation
8. Clear gradients with optimizer.zero_grad(set_to_none=True) for efficiency
