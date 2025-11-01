# nanoGPT2 🧠

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A minimal-yet-practical GPT-2 style transformer implemented from scratch for learning, research, and experimentation. Clean, readable code with comments and tests—focused on understanding how modern LLMs work end-to-end.

## ✨ Goals
- Pedagogical clarity: each component is small and well-documented
- Practical training: train on small datasets on a single GPU/CPU
- Reproducible results: seeds, configs, and deterministic dataloaders
- Extensible: easy to add new blocks, attention variants, and schedulers

## 🧩 Features
- Tokenizer (BPE or byte-level)
- Transformer blocks (MHSA, MLP, LayerNorm, residuals)
- Causal attention with attention masking
- Weight tying between embeddings and output head
- Configurable model sizes (nano, small, base)
- Trainer with gradient clipping, mixed precision (optional)
- Cosine LR scheduler with warmup
- Checkpointing and resume support

## 📦 Installation
```bash
git clone https://github.com/d-negatu/nanoGPT2.git
cd nanoGPT2
pip install -e .
```

## 🚀 Quickstart
Train a tiny model on tiny Shakespeare:
```bash
python scripts/prepare_shakespeare.py
python train.py --config configs/nano.yaml
```

Sample text:
```bash
python sample.py --checkpoint runs/nano/latest.pt --max-new-tokens 200 --temperature 0.9
```

## 🛠️ Project Structure
```
nanoGPT2/
├── gpt2/
│   ├── model.py          # Transformer model
│   ├── blocks.py         # Attention + MLP blocks
│   ├── tokenizer.py      # Byte-level or BPE tokenizer
│   └── utils.py          # Positional encodings, masks, etc.
├── scripts/
│   ├── prepare_shakespeare.py
│   └── prepare_openwebtext.py
├── configs/
│   ├── nano.yaml         # ~1-5M params
│   ├── small.yaml        # ~10-30M params
│   └── base.yaml         # ~100M params
├── train.py              # Training loop
├── sample.py             # Sampling/generation
├── tests/                # Unit tests
└── README.md
```

## ⚙️ Example Config (nano.yaml)
```yaml
model:
  vocab_size: 50304
  n_layer: 6
  n_head: 6
  n_embd: 384
  dropout: 0.1
train:
  seq_len: 256
  batch_size: 64
  max_steps: 10000
  lr: 3e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  warmup_steps: 200
  grad_clip: 1.0
  amp: false
```

## 📈 Tips for Good Results
- Start with seq_len=128 to iterate quickly, then increase
- Use cosine decay with warmup for stable convergence
- Enable AMP on GPUs for 1.5-2x speedup
- Keep attention heads divisible by embedding dim

## 🧪 Testing
```bash
pytest -q
```

## 🔭 Roadmap
- [ ] Rotary embeddings (RoPE)
- [ ] FlashAttention (fallback implementation)
- [ ] LoRA fine-tuning
- [ ] Export to ONNX
- [ ] Web demo with Gradio

## 📚 References
- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- GPT-2 open-source reimplementations

## 👤 Author
Dagmawi Negatu — Western Carolina University
- GitHub: https://github.com/d-negatu
- LinkedIn: https://www.linkedin.com/in/danegatu

If this repo helps you learn, please ⭐ it to support the project!
