# Overview

The main goal is to understand some of the methods used for token pruning in VLM, specially when applicated to autonomous-driving scenarios.

Transformers can naturally process sequences of different lengths. Which makes easier to apply pruning techniques.

# Papers

- [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461)
  - Token merging, rather than token reduction.
  - The paper places this merging between the attention branch and the MLP branch of each transformer block.
  - Can be used both in training or inference.
  - Use a dot product similarity metric between the keys of each token to determine which contain similar information.
  - Bipartite Soft Matching algorithm decides which tokens to merge.
  - Use of proportional attention to represent merged tokens as more relevant than a single token.

- [An Image is Worth 1/2 Tokens After Layer 2: Plug-and-PLay Acceleration for VLLM Inference](https://arxiv.org/abs/2403.06764)
  - FastV explores visual token reduction for Large Vision-Language Models (LVLMs).
  - "Another significant advantage of FastV over previous methods is its simplicity; it can be applied to any LVLM without requiring model retraining."
  - Compute "attention efficiency" to evaluate if image tokens are being relevant to the model.
  - Prune least efficient tokents.

- [Token Pruning in Multimodal Large Language Models: Are we solving the right problem?](https://arxiv.org/abs/2502.11501)
  - Criticizes existing pruning methods and their evaluation procedures. They can be worst than random or pooling techniques in some datasets. The pruning algorithms themselves have a computational cost that can make the final computational power consumption remain equal. Finallym they do not consider "token duplication alongside token importance".
  - Fastv suffer from poision bias during token reduction.
  - Various token pruning methods exhibit a significant degradation in performance when applied to precise obvject localization.
  - Considering both predictive criticality and uniqueness to decide whether to keep a token or not.

- [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](https://arxiv.org/abs/2410.04417)

- [Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](https://arxiv.org/abs/2412.13180)
  - FastV has problems. The visual encodings are not being properly evaluated and it is really limited regarding localization demanding tasks.
  - Shorter distance information is considered more valuable in LM, which makes sense for text, but not for vision.
  - Do not apply RoPE when pruning,
  - How can we encode positional information in visually conditioned language models to prevent positional artifacts in cross-modal interactions?

- [DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models](https://arxiv.org/abs/2503.02175) 
  - We introduce DivPrune, a token pruning method based on MMDP that maximizes diversity among visual tokens, effectively reducing redundancy and ensuring a highly representative subset. 
  - Find a set that has the maximum minimum distance between its elements. This distance is defined using cosine distance.
  - This proves that redundancy of visual tokens leads to poor performance and diversity maximization is needed for optimal performance.

- [Prune2Drive](https://arxiv.org/abs/2508.13305)
  - One of the first to suggest a multi-view aware technique. It has two steps: Diversity-aware token pruning and View-adaptive pruning ratio optimization.
  - Inspired by Farthest Point Sampling (FPS), chooses a set of tokens whoose min internal distance (cosine similarity) is maximized.
  - Find a retention ratio $\alpha$ for each view of the camera.