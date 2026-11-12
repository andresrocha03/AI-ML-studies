# Transformers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Transformers is an attention based network architecture, introduced in 2017. Its advantages are to capture context better than previous architectures, such as recurrences or convolutions.

# Visual Transformers

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)

This paper introduced transformers applied to image recognition by turning an image into sevral patches which are fed into the transformer as tokens. 

## ViT Variants

### Swin Transformer
- Uses **local self-attention inside windows** instead of global attention.
- Alternates with **shifted windows** so information can move across neighboring regions.
- Uses **patch merging** between stages to reduce the number of tokens and increase the feature dimension.
- Has a hierarchical structure similar to CNNs.

### MViT
- Uses a **multiscale hierarchical architecture**.
- Progressively reduces the number of spatial tokens through **pooling**.
- Increases feature richness as the network gets deeper.
- Its main efficiency mechanism is reducing the resolution of the representations used in attention.

### LeViT
- Hybrid architecture combining **CNN operations and Transformer attention**.
- Uses convolutions for efficient early feature extraction and downsampling.
- Progressively reduces spatial resolution while increasing feature dimension.
- Designed mainly for **fast inference and a good accuracy/latency tradeoff**.


## ViT Training Techniques

| Technique | Type | Main Idea |
|---|---|---|
| **DeiT** | Supervised | Trains ViT efficiently on ImageNet using strong augmentation/regularization. Can also use a pretrained **teacher model** for knowledge distillation. |
| **AugReg** | Supervised | Uses strong **data augmentation and regularization** to improve ViT generalization. |
| **MAE** | Self-supervised | Masks a large fraction of image patches and trains the ViT to **reconstruct the missing patches**. |
| **SWAG** | Weakly supervised | Pretrains ViT on a very large dataset of **weak/noisy labels**, then fine-tunes it for downstream tasks. |