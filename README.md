# nanoGPT2 🧠

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-negatu/nanoGPT2/blob/main/notebooks/Quick_Start_nanoGPT2.ipynb)
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

## 🎬 Demo

![nanoGPT2 Demo](docs/demo.svg)

Quick demonstration of nanoGPT2 capabilities:
- Minimal GPT model with 4 layers
- Train on custom text datasets
- Generate text with a simple API
- Run on GPU or CPU
- Interactive Colab notebook available

### Try it Now

Get started immediately with our interactive Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-negatu/nanoGPT2/blob/main/notebooks/Quick_Start_nanoGPT2.ipynb)

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
├── src/
│   ├── model.py          # Transformer model
│   ├── blocks.py         # Attention + MLP blocks
│   ├── tokenizer.py      # Byte-level or BPE tokenizer
│   ├── trainer.py        # Training loop
│   └── utils.py          # Positional encodings, masks, etc.
├── notebooks/
│   └── Quick_Start_nanoGPT2.ipynb  # Interactive Colab notebook
├── scripts/
│   ├── prepare_shakespeare.py
│   └── demo.py           # Generate demo outputs
├── docs/
│   ├── demo.svg          # Demo visualization
│   ├── architecture.md    # Model architecture details
│   ├── training.md        # Training guide
│   └── api.md             # API reference
├── configs/
│   ├── nano.yaml         # Smallest config
│   ├── small.yaml        # Small config
│   └── base.yaml         # Base config
├── train.py              # Main training script
├── sample.py             # Sampling script
├── CHANGELOG.md          # Release notes and changelog
└── README.md             # This file
```

## 📚 Documentation

For more detailed information, check out our documentation:

- **[Architecture Guide](docs/architecture.md)** - Deep dive into the model architecture
- **[Training Guide](docs/training.md)** - Tips and tricks for training your own models
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Changelog](CHANGELOG.md)** - Release notes and feature history
- **[Interactive Notebook](notebooks/Quick_Start_nanoGPT2.ipynb)** - Try it on Colab

## 💡 Usage Examples

### Basic Training

```python
from src.model import GPT
from src.trainer import Trainer

config = {'vocab_size': 256, 'block_size': 128, 'n_layer': 4}
model = GPT(config)
trainer = Trainer(model, config)
trainer.train()
```

### Text Generation

```python
import torch

context = torch.zeros((1, 1), dtype=torch.long)
samples = model.generate(context, max_new_tokens=100)
print(samples)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues and submit pull requests.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and educational resources on transformers.
