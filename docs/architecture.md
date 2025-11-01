# nanoGPT2 Architecture

## Overview

nanoGPT2 is a minimal yet complete implementation of a GPT-2 style transformer architecture. This document provides a detailed explanation of each component.

## Core Components

### 1. Token Embeddings

- **Vocabulary Embeddings**: Maps token indices to dense vectors of size `n_embd`
- **Positional Embeddings**: Adds position-dependent information to each token
- Learnable embeddings optimized during training

### 2. Transformer Blocks

Each transformer block consists of:

#### Multi-Head Self-Attention (MHSA)
- Splits embedding into `n_head` attention heads
- Each head has dimension `n_embd // n_head`
- Computes query, key, value projections
- Applies scaled dot-product attention with causal masking
- Concatenates heads and projects back to `n_embd` dimensions

#### Feed-Forward Network (FFN)
- Two-layer MLP with GELU activation
- Expands to `4 * n_embd` dimensions, then projects back
- Provides non-linearity and model expressiveness

#### Layer Normalization & Residuals
- Pre-layer normalization before each sub-layer
- Residual connections enable deep networks
- Improves gradient flow during backpropagation

### 3. Output Head

- Projects final transformer output to vocabulary size
- Generates probability distribution over next tokens
- Optionally shares weights with input embeddings (weight tying)

## Configuration Parameters

| Parameter | Description | Default (nano) |
|-----------|-------------|----------------|
| `vocab_size` | Number of unique tokens | 256 (byte-level) |
| `block_size` | Context length / max sequence length | 128 |
| `n_layer` | Number of transformer blocks | 4 |
| `n_head` | Number of attention heads | 4 |
| `n_embd` | Embedding dimension | 64 |
| `dropout` | Dropout rate | 0.0 |
| `bias` | Use bias in linear layers | True |

## Forward Pass

1. Input token indices (shape: `[batch_size, block_size]`)
2. Token embeddings + positional embeddings
3. Apply dropout
4. Process through N transformer blocks
5. Apply layer normalization
6. Project to vocabulary logits
7. Output logits (shape: `[batch_size, block_size, vocab_size]`)

## Attention Mechanism

Causal self-attention ensures that predictions only use previous tokens:

```
Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k) + mask) @ V
```

- **Causal Mask**: Lower triangular matrix (future tokens masked)
- **Scaling**: By sqrt(d_k) for stability
- **Softmax**: Converts scores to probability distribution

## Model Sizes

nanoGPT2 supports multiple configuration sizes:

### Nano (Default)
- 4 layers, 4 heads, 64 embedding dimension
- ~250K parameters
- Trains quickly on CPU/single GPU

### Small
- 8 layers, 8 heads, 256 embedding dimension
- ~2-3M parameters

### Base
- 12 layers, 12 heads, 768 embedding dimension
- ~100M+ parameters
- Comparable to small GPT-2

## Training Details

### Loss Function
Cross-entropy loss computed over all positions in the sequence:

```
Loss = -1/N * sum(log(p_true))
```

### Optimization
- Adam optimizer with configurable learning rate
- Cosine learning rate scheduler with warmup
- Gradient clipping for stability
- Mixed precision (optional) with automatic mixed precision (AMP)

### Regularization
- Dropout in embedding and transformer layers
- Weight decay (L2 regularization)
- Layer normalization for internal covariate shift

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpuksxwoqaxqos7qaqga5id3lis4ctntw57mvjqaxvliifkq.ipfs.dweb.link/ipfs/bafybeiepqvhb643e7pejpfq6bpxkwbpqpglrh5v3cfcslsva24jxblwpdi/1902.03592.pdf) - Radford et al., 2019 (GPT-2 paper)

## Future Enhancements

- [ ] Rotary positional embeddings (RoPE)
- [ ] Multi-query attention (MQA)
- [ ] Grouped query attention (GQA)
- [ ] Flash attention kernels
- [ ] Quantization support
