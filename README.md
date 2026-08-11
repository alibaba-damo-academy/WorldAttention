<p align="center">
  <img width="1915" height="657" alt="worldattention_crop" src="https://github.com/user-attachments/assets/8f629318-1e95-4c02-ac08-59a451d2316c" />
</p>

<h1 align="center">WorldAttention: An Efficient Attention Architecture for Interactive Video World Models</h1>

<!-- <h3 align="center">2026</h3> -->

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="#"><img src="https://img.shields.io/badge/HF_Paper-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Dataset"></a>
</p>

<p align="center">
  <a href="https://steve-zeyu-zhang.github.io/">Zeyu Zhang</a><sup>1*</sup> &nbsp;
  <a href="https://github.com/skeletalknight">Jinyuan Mao</a><sup>2*</sup> &nbsp;
  <a href="https://github.com/andakai">Dakai An</a><sup>3*</sup> &nbsp;
  <a href="https://wangbo-zhao.github.io/">Wangbo Zhao</a><sup>3</sup> &nbsp;
  <a href="https://scholar.google.com/citations?user=QJGroC0AAAAJ&hl=en">Hanfeng Lu</a><sup>3</sup> &nbsp;
  <a href="https://openreview.net/profile?id=%7EJiasheng_Tang1">Jiasheng Tang</a><sup>1&#8224;</sup> &nbsp;
  <a href="https://scholar.google.com/citations?user=_YgjRn0AAAAJ&hl=en">Yinghao Yu</a><sup>4</sup> &nbsp;
  <a href="https://www.cse.ust.hk/~weiwa/">Wei Wang</a><sup>3</sup> &nbsp;
  <a href="https://bohanzhuang.github.io/">Bohan Zhuang</a><sup>1,2&#8224;</sup>
</p>

<p align="center">
  <sup>1</sup>DAMO Academy, Alibaba Group &nbsp;&nbsp;
  <sup>2</sup>Zhejiang University &nbsp;&nbsp;
  <sup>3</sup>HKUST &nbsp;&nbsp;
  <sup>4</sup>TRE, Alibaba Group
</p>

<p align="center">
  <sup>*</sup>Equal contribution. &nbsp;&nbsp;
  <sup>&#8224;</sup>Corresponding authors.
</p>

<p align="center">
  <img width="2900" height="1121" alt="Figure-1-1" src="https://github.com/user-attachments/assets/b33b8930-c7e6-4027-b094-0821fd8f4c73" />
</p>

_We propose a system-oriented co-design using Hierarchical KV Cache (HKV) for coarse-to-fine memory retrieval, and Hybrid Sparse Attention (HSA) for efficient dual-branch attention computation._

## To-Do List

- [x] Release the paper on arXiv
- [x] Build the project page
- [x] Release the core code
- [ ] Release the inference pipeline
- [ ] Release the HSA kernels
- [ ] Release the visualization script
- [ ] Release the demo

## Intro

Leveraging the paradigm of autoregressive diffusion, text-conditioned interactive video world models
aim to simulate temporally coherent environments guided by textual instructions. While enabling
low-latency, long-duration generation is pivotal for embodied AI and simulation-based planning,
current frameworks primarily rely on sliding-window mechanisms to bound computational complexity.
However, this approach inherently sacrifices historical context, undermining the long-range
interactive capabilities. Conversely, maintaining a full-history cache remains computationally
prohibitive and memory-intensive: the quadratic complexity of attention leads to excessive
computational overhead, while the linear growth of the KV cache inevitably leads to GPU memory
saturation. To overcome these limitations, we propose **WorldAttention**, a system-aware attention
architecture that achieves high efficiency through the co-design of specialized attention kernels
and hierarchical KV cache management. First, we introduce **Hybrid Sparse Attention (HSA)**, which
integrates linear global attention supplemented with head-adaptive sparse attention. Additionally,
we design a **Hierarchical KV Cache (HKV)** that organizes historical KV pairs into semantically
indexed pages across multi-tier memory, enabling fine-grained retrieval and controlled GPU
residency. These two designs are supported by tailored kernels to effectively translate their
theoretical efficiency into real-world performance. Extensive experiments on VBench-Long and
InterVBench demonstrate that WorldAttention consistently surpasses prior state-of-the-art methods,
achieving subject consistency scores of 0.9472 on VBench-Long and 0.9668 on InterVBench,
respectively. Our system-oriented kernel design for HSA brings a 14.02× speedup over
FlashAttention-3, and a 2.21× end-to-end speed up together with HKV. At inference, WorldAttention
sustains 22.0 FPS on a single NVIDIA H100.

## News

## Quick Start

### Environment

```bash
git clone https://github.com/alibaba-damo-academy/WorldAttention.git
cd WorldAttention

conda create -n worldattention python=3.10 -y
conda activate worldattention
pip install -r requirements.txt
```

`triton` is optional. It enables the Triton block-sparse backend for the HSA sparse branch.

### Training

Three stages, each resuming from the previous one's checkpoint:

```bash
# Stage 1 - adapt the bidirectional model into a 4-step causal student (dense attention)
torchrun --nproc_per_node=8 train.py \
  --config_path configs/train_stage1_causal.yaml --logdir logs_stage1

# Stage 2 - HSA warmup: base frozen, HSA parameters trained by attention self-distillation
torchrun --nproc_per_node=8 train.py \
  --config_path configs/train_stage2_hsa_warmup.yaml --logdir logs_stage2

# Stage 3 - HSA tune: streaming prompt-switch distillation with HSA active
torchrun --nproc_per_node=8 train.py \
  --config_path configs/train_stage3_hsa_tune.yaml --logdir logs_stage3
```

Model weights are read from `models/` by default, and prompt lists from `prompts/`.

## Citation

```bibtex
@article{zhang2026worldattention,
  title   = {WorldAttention: An Efficient Attention Architecture for Interactive Video World Models},
  author  = {Zhang, Zeyu and Mao, Jinyuan and An, Dakai and Zhao, Wangbo and Lu, Hanfeng and
             Tang, Jiasheng and Yu, Yinghao and Wang, Wei and Zhuang, Bohan},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
