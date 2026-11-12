# Failure modes

# Papers

- [Visual Information Shift](https://arxiv.org/abs/2604.12358)
    - Focused on visual math reasoning tasks.
    - Model's visual focues does not remain fixed as reasoning progresses.
    - Adaptation to make existing pruning methods such as FastV and DivPrune, more robust to these shifts.
    
- [Token Pruning in Multimodal Large Language Models: Are we solving the right problem?](https://arxiv.org/abs/2502.11501)
  - Criticizes existing pruning methods and their evaluation procedures. They can be worst than random or pooling techniques in some datasets. The pruning algorithms themselves have a computational cost that can make the final computational power consumption remain equal. Finallym they do not consider "token duplication alongside token importance".
  - Fastv suffer from poision bias during token reduction.
  - Various token pruning methods exhibit a significant degradation in performance when applied to precise obvject localization.
  - Considering both predictive criticality and uniqueness to decide whether to keep a token or not.
