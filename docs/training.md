# Training Guide

This guide covers how to train nanoGPT2 models on your own datasets.

## Preparing Your Data

### Format
Data should be in plain text format (.txt files). The model uses character-level tokenization by default.

### Creating a Dataset

```python
from src.dataset import CharDataset

# Load your text file
with open('data/mytext.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create dataset
dataset = CharDataset(text, block_size=128)
```

## Basic Training

### Using Configuration Files

Train with a preset configuration:

```bash
python train.py --config configs/nano.yaml
```

### Configuration Parameters

```yaml
# Model architecture
vocab_size: 256          # For byte-level tokenization
block_size: 128          # Context length
n_layer: 4               # Number of transformer blocks
n_head: 4                # Number of attention heads
n_embd: 64               # Embedding dimension
dropout: 0.1             # Dropout rate

# Training settings
batch_size: 64           # Batch size
learning_rate: 1e-3      # Learning rate
max_iters: 5000          # Total training iterations
eval_iters: 100          # Evaluation iterations
eval_interval: 500       # Evaluate every N iterations

# Hardware
device: 'cuda'           # 'cuda' or 'cpu'
mixed_precision: false   # Use AMP if true

# Checkpointing
save_checkpoint: true
checkpoint_interval: 1000
checkpoint_dir: './checkpoints'
```

### Programmatic Training

```python
import torch
from src.model import GPT
from src.trainer import Trainer
from src.config import get_config

# Get configuration
config = get_config('nano')

# Initialize model
model = GPT(config)

# Create trainer
trainer = Trainer(model, config)

# Train
trainer.train(train_dataset, val_dataset)

# Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'checkpoints/final_model.pt')
```

## Monitoring Training

The trainer will print:
- Training loss
- Validation loss
- Tokens per second
- Elapsed time

```
iter 100: train loss 4.1234, val loss 4.1567 (time: 23.45s)
iter 200: train loss 3.8765, val loss 3.9123 (time: 47.89s)
```

## Tips for Better Training

### Learning Rate
- Start with `1e-3` for most datasets
- Reduce if training is unstable (loss oscillating)
- Increase if training is too slow

### Batch Size
- Larger batches are more stable but require more GPU memory
- Typical range: 32-128
- Adjust based on your GPU/CPU memory

### Context Length
- Longer sequences capture more context but are slower to train
- For small datasets: 128-256
- For large datasets: 512-1024+

### Model Size
- **Nano** (default): Fast training, good for experimentation
- **Small**: Balanced performance and speed
- **Base**: Better quality but slower

### Data Preparation
- Use high-quality, clean text
- Remove formatting artifacts
- Combine multiple sources if needed
- Minimum recommended: 1MB of text

### Hardware Acceleration
- Use GPU for faster training (10-50x speedup)
- Enable mixed precision for additional speedup with minimal quality loss
- CPU training is feasible for experimentation

## Training on Colab

See [Quick_Start_nanoGPT2.ipynb](../notebooks/Quick_Start_nanoGPT2.ipynb) for a complete example notebook.

## Common Issues

### Out of Memory (OOM)
- Reduce `batch_size`
- Reduce `block_size`
- Use a smaller model config

### Loss not decreasing
- Check data format
- Verify learning rate (may be too high)
- Ensure sufficient training iterations

### Very slow training
- Use GPU instead of CPU
- Reduce `block_size`
- Reduce `batch_size` to parallelize better

## Next Steps

- Try different model sizes
- Experiment with hyperparameters
- Train on different datasets
- Fine-tune a pretrained model

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Transformer Papers](https://github.com/d-negatu/nanoGPT2/docs/architecture.md)
