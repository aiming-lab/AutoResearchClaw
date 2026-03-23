---
name: cv-detection
description: Best practices for object detection tasks. Use when working on COCO, VOC, or detection architectures like YOLO and DETR.
metadata:
  category: domain
  trigger-keywords: "detection,object,bbox,yolo,coco,anchor,faster rcnn"
  applicable-stages: "9,10"
  priority: "5"
  version: "1.0"
  author: researchclaw
  references: "Ren et al., Faster R-CNN, NeurIPS 2015; Carion et al., End-to-End Object Detection with Transformers, ECCV 2020"
---

## Object Detection Best Practice
Architecture families:
- One-stage: YOLO (v5/v8), SSD, RetinaNet, FCOS
- Two-stage: Faster R-CNN, Cascade R-CNN
- Transformer: DETR, DINO, RT-DETR

Training recipe:
- Use pre-trained backbone (ImageNet)
- Multi-scale training and testing
- IoU threshold: 0.5 for mAP50, 0.5:0.95 for mAP
- Use FPN for multi-scale feature extraction
- Focal loss for class imbalance in one-stage detectors

Standard benchmarks:
- COCO val2017: ~37 mAP (Faster R-CNN R50), ~51 mAP (DINO Swin-L)
- Pascal VOC: ~80 mAP50 (Faster R-CNN)
