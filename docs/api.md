# API Reference

Complete API documentation for nanoGPT2.

## Model Classes

### GPT

Main model class implementing a GPT-2 style transformer.

```python
from src.model import GPT

class GPT:
    """GPT-2 style transformer model"""
```

#### Constructor

```python
GPT(config: dict)
```

**Parameters:**
- `config` (dict): Configuration dictionary containing:
  - `vocab_size` (int): Vocabulary size
  - `block_size` (int): Maximum sequence length
  - `n_layer` (int): Number of transformer layers
  - `n_head` (int): Number of attention heads
  - `n_embd` (int): Embedding dimension
  - `dropout` (float): Dropout rate
  - `bias` (bool): Whether to use bias in linear layers

#### Methods

##### forward()

```python
model.forward(idx, targets=None)
```

Forward pass through the model.

**Parameters:**
- `idx` (torch.Tensor): Input token indices, shape `[batch_size, seq_len]`
- `targets` (torch.Tensor, optional): Target token indices for loss computation

**Returns:**
- `logits` (torch.Tensor): Output logits, shape `[batch_size, seq_len, vocab_size]`
- `loss` (torch.Tensor, optional): Cross-entropy loss if targets provided

**Example:**
```python
import torch

# Forward pass without targets
logits = model(idx)

# Forward pass with targets (training)
logits, loss = model(idx, targets)
```

##### generate()

```python
model.generate(idx, max_new_tokens, temperature=1.0, top_k=None)
```

Generate new tokens autoregressively.

**Parameters:**
- `idx` (torch.Tensor): Context token indices, shape `[batch_size, seq_len]`
- `max_new_tokens` (int): Number of new tokens to generate
- `temperature` (float): Sampling temperature (>0). Higher = more random
- `top_k` (int, optional): If set, only sample from top-k most likely tokens

**Returns:**
- `torch.Tensor`: Generated token indices including context, shape `[batch_size, seq_len + max_new_tokens]`

**Example:**
```python
# Generate from empty context
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=100, temperature=0.9)

# Generate with top-k sampling
generated = model.generate(context, max_new_tokens=100, top_k=50)
```

## Trainer Class

### Trainer

Handles model training, validation, and checkpointing.

```python
from src.trainer import Trainer

class Trainer:
    """Training orchestrator for GPT models"""
```

#### Constructor

```python
Trainer(model, config)
```

**Parameters:**
- `model` (GPT): Model instance to train
- `config` (dict): Configuration dictionary

#### Methods

##### train()

```python
trainer.train(train_dataset=None, val_dataset=None)
```

Train the model.

**Parameters:**
- `train_dataset` (Dataset, optional): Training dataset
- `val_dataset` (Dataset, optional): Validation dataset

**Example:**
```python
trainer = Trainer(model, config)
trainer.train(train_data, val_data)
```

## Dataset Class

### CharDataset

Character-level dataset for text data.

```python
from src.dataset import CharDataset

class CharDataset:
    """Character-level text dataset"""
```

#### Constructor

```python
CharDataset(text, block_size)
```

**Parameters:**
- `text` (str): Input text
- `block_size` (int): Context length / max sequence length

**Example:**
```python
with open('data/text.txt', 'r') as f:
    text = f.read()

dataset = CharDataset(text, block_size=128)
```

#### Methods

##### __len__()

Get dataset size.

```python
len(dataset)  # Returns number of sequences
```

##### __getitem__()

Get a single sample.

```python
idx, targets = dataset[0]
```

**Returns:**
- `idx` (torch.Tensor): Input token indices
- `targets` (torch.Tensor): Target token indices (shifted by 1)

## Utilities

### Config Utilities

```python
from src.config import get_config

# Get preset configuration
config = get_config('nano')    # 'nano', 'small', or 'base'

# Modify configuration
config['learning_rate'] = 5e-4
```

### Device Utilities

```python
import torch

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model.to(device)
```

## Complete Training Example

```python
import torch
from src.model import GPT
from src.trainer import Trainer
from src.dataset import CharDataset
from src.config import get_config

# Load data
with open('data/input.txt', 'r') as f:
    text = f.read()

# Create dataset
dataset = CharDataset(text, block_size=128)

# Split into train/val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Create model
config = get_config('nano')
model = GPT(config)

# Train
trainer = Trainer(model, config)
trainer.train(train_dataset, val_dataset)

# Generate
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=100)
print(generated)
```

## Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: CUDA out of memory` | GPU memory exceeded | Reduce batch_size or block_size |
| `IndexError: vocab index out of range` | Invalid token indices | Check tokenizer output |
| `TypeError: expected Tensor` | Wrong input type | Ensure inputs are torch.Tensor |

