# Benchmarks failure modes


### Papers

- [A Survey of State of the Art Large Vision Language Models: Alignment,Benchmark, Evaluations and Challenges](https://arxiv.org/abs/2501.02189)
    - This paper talks about different aspects of VLMs. Lets focus on Benchmarks here.
    - Synthetic generated questions datasets are a common alternative for scaling training data.
        - The images are real, but they generate captions and use this captions for generating questions.
        - They could also provide images with examples of questions for a LLM to generate more adversarial questions.
        - The questions benig created may not require visual reasoning.

    - Evaluation Metrics:     
        - Most of them rely on multiple-choice format or yes/no questions.
    
    - VLM Failures:
        - Hallucination: referencing nonexistent objects in images.
        - Safety: people could try to cause harm by attacking VLM applications.

- [Are We on the Right Way for Evaluating Large Vision-Language Models?](https://arxiv.org/abs/2403.20330)
  - Exiting benchmarks are poorly evaluating models.
  - Visual content is unnecessary for many samples.
  - Unintentional data leakage in the training of LLM and LVLM.
  - Every benchmark shows a certain degree of samples that visual contents are unnecessary.