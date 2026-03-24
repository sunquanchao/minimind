# MiniMind Project Introduction and Architecture

## 📋 Project Overview

**MiniMind** is a from-scratch Large Language Model (LLM) training framework that implements all core algorithms using native PyTorch code. The project aims to lower the barrier to LLM learning, allowing developers to deeply understand the internal mechanisms of models rather than just calling highly abstract APIs.

### Core Philosophy

- **From-Scratch Implementation**: All core algorithms (Attention, FFN, RoPE, MoE, PPO, DPO, etc.) are reimplemented using native PyTorch code
- **Minimalist Design**: Smallest model has only 26M parameters, can be trained quickly on consumer GPUs
- **Full Pipeline Coverage**: Covers complete training workflow from data cleaning, pretraining, SFT, LoRA, RLHF to model distillation
- **Low Entry Barrier**: Extremely low cost (~3 CNY + 2 hours), suitable for learning and research

### Key Features

| Feature | Description |
|---------|-------------|
| **Ultra-Small** | MiniMind2-Small has only 26M parameters, 1/7000 of GPT-3 |
| **Fast Training** | ~2 hours on single RTX 3090 |
| **Low Cost** | GPU server rental cost ~3 CNY |
| **Multimodal Support** | Extended version MiniMind-V supports vision-language tasks |
| **Full-Stack Implementation** | No dependence on transformers+trl abstract interfaces |

## 🏗️ System Architecture

### Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        MiniMind Training Pipeline               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Tokenizer│ -> │ Pretrain │ -> │   SFT    │ -> │  RLHF    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                  │               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ↓               │
│  │ Dataset  │ -> │  Model   │ -> │ Trainer  │ ┌──────────┐      │
│  └──────────┘    └──────────┘    └──────────┘ │ Inference│      │
│                                              └──────────┘      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
minimind/
├── model/                    # Model architecture definitions
│   └── model_minimind.py    # MiniMind model core implementation
│       ├── MiniMindConfig   # Model configuration class
│       ├── Attention        # Multi-head attention mechanism
│       ├── FeedForward      # Feed-forward neural network
│       ├── MoEGate          # MoE gating mechanism
│       └── MiniMindForCausalLM  # Complete causal language model
├── dataset/                  # Dataset processing
│   └── lm_dataset.py        # Dataset class definitions
│       ├── PretrainDataset  # Pretraining dataset
│       ├── SFTDataset       # Supervised fine-tuning dataset
│       ├── DPODataset       # Preference optimization dataset
│       └── RLAIFDataset     # RLHF training dataset
├── trainer/                  # Training scripts
│   ├── train_tokenizer.py   # Tokenizer training
│   ├── train_pretrain.py    # Pretraining
│   ├── train_full_sft.py    # Full SFT
│   ├── train_lora.py        # LoRA fine-tuning
│   ├── train_dpo.py         # DPO training
│   ├── train_ppo.py         # PPO training
│   └── train_reason.py      # Reasoning model training
├── scripts/                  # Utility scripts
│   ├── web_demo.py          # Web UI demo
│   └── serve_openai_api.py  # OpenAI-compatible API
├── eval_llm.py              # Model inference
└── requirements.txt         # Dependency list
```

## 🔧 Core Modules

### 1. Model Architecture (model/model_minimind.py)

#### 1.1 MiniMindConfig

Model configuration class inheriting from `PretrainedConfig`, supporting flexible model scaling:

```python
# Basic configuration
- hidden_size: 512                # Hidden layer dimension
- num_hidden_layers: 8            # Number of transformer layers
- num_attention_heads: 8          # Number of attention heads
- num_key_value_heads: 2          # GQA KV heads
- vocab_size: 6400                # Vocabulary size
- max_position_embeddings: 32768  # Maximum sequence length

# MoE configuration (optional)
- use_moe: bool                   # Enable MoE
- n_routed_experts: 4             # Total number of routed experts
- num_experts_per_tok: 2          # Experts selected per token
- n_shared_experts: 1             # Number of shared experts
```

#### 1.2 Attention Mechanism

Implements multi-head attention mechanism with support for:
- **GQA (Grouped Query Attention)**: Reduces KV cache memory usage
- **RoPE (Rotary Position Embedding)**: Rotary position encoding
- **YaRN Extrapolation**: Supports ultra-long sequences (up to 32K)
- **Flash Attention**: Optional efficient attention implementation
- **KV Cache**: Inference acceleration

```python
class Attention(nn.Module):
    # Q: All heads
    # K/V: Only KV heads (expanded via repeat_kv)
    # RoPE: Rotary position encoding
    # Flash Attention: Optional acceleration
```

#### 1.3 Feed-Forward Network

**Standard FFN (FeedForward)**:
- SwiGLU activation function
- Gating mechanism: `act_fn(gate_proj(x)) * up_proj(x)`

**MoE FFN (MOEFeedForward)**:
```python
class MOEFeedForward(nn.Module):
    - MoEGate: Soft routing gating mechanism
    - Expert selection: Top-K routing
    - Load balancing: Auxiliary loss (aux_loss)
    - Shared experts: Shared by all tokens
```

#### 1.4 Transformer Block

Uses Pre-LN architecture:
```
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

#### 1.5 Model Specifications

| Model | Parameters | hidden_size | layers | heads | Use Case |
|-------|------------|-------------|--------|-------|----------|
| MiniMind2-Small | 26M | 512 | 8 | 8/2 | Lightweight dialogue |
| MiniMind2-Base | 104M | 768 | 12 | 12/4 | Standard dialogue |
| MiniMind2-MoE | 145M | 768 | 12 | 12/4 | High-performance inference |

### 2. Dataset Processing (dataset/lm_dataset.py)

#### 2.1 PretrainDataset

Pretraining dataset for learning language patterns from raw text:
- Input: JSON format `{text: "..."}`
- Processing: Add BOS/EOS tokens, pad to max_length
- Labels: All tokens participate in training

#### 2.2 SFTDataset

Supervised fine-tuning dataset for instruction-following training:
- Input: Multi-turn dialogue format `[{role, content}, ...]`
- Processing:
  - Supports multi-turn conversations
  - Auto-add system prompt (20% probability)
  - Supports function calling (tools)
- Labels: **Only assistant responses** participate in training

#### 2.3 DPODataset

Direct Preference Optimization dataset:
- Input: chosen and rejected dialogue pairs
- Output: prompt + chosen/rejected response
- Loss mask: Calculate loss only on assistant responses

#### 2.4 RLAIFDataset

RLHF training dataset (PPO/GRPO):
- Input: List of prompts
- Output: Prompts for reinforcement learning generation

### 3. Training Pipeline (trainer/)

#### 3.1 Training Stages

```
1. Tokenizer Training
   ↓
2. Pretraining
   Learn basic language capabilities from random initialization
   ↓
3. SFT (Supervised Fine-Tuning)
   Align model to understand instructions and learn dialogue capabilities
   ↓
4. Optional: LoRA Fine-Tuning
   Efficient parameter fine-tuning for domain adaptation
   ↓
5. Optional: RLHF
   - DPO: Direct Preference Optimization
   - PPO/GRPO/SPO: Policy gradient optimization
   ↓
6. Optional: Reasoning
   Train reasoning capabilities (using ｜think＞ tags)
```

#### 3.2 Training Configuration

```python
# Distributed training
- DDP (DistributedDataParallel) support
- Gradient accumulation (accumulation_steps)
- Mixed precision (bfloat16/float16)
- torch.compile acceleration (optional)

# Checkpoint saving
- Model weights: ../out/{weight}_{hidden_size}.pth
- Full state: ../checkpoints/{weight}_{hidden_size}_resume.pth
- Auto-resume: --from_resume 1

# Experiment tracking
- WandB integration: --use_wandb
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Train Tokenizer

```bash
python -m trainer.train_tokenizer
```

### 3. Pretraining

```bash
python -m trainer.train_pretrain \
    --data_path ../dataset/pretrain_hq.jsonl \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 340
```

### 4. Supervised Fine-Tuning

```bash
python -m trainer.train_full_sft \
    --from_weight pretrain \
    --data_path ../dataset/sft_mini.jsonl \
    --epochs 2 \
    --batch_size 64 \
    --learning_rate 5e-5
```

### 5. Model Inference

```bash
# Interactive dialogue
python eval_llm.py

# Web UI
streamlit run scripts/web_demo.py

# API service
python scripts/serve_openai_api.py
```

## 💡 Technical Highlights

### 1. From-Scratch Core Algorithms

- **Attention**: Manual implementation of QKV projection, RoPE, Flash Attention
- **FFN/MoE**: Native implementation of SwiGLU, expert routing, load balancing
- **RoPE + YaRN**: Manual implementation of position encoding and extrapolation
- **RLHF**: Manual implementation of PPO, GRPO, DPO, SPO algorithms

### 2. Efficient Design

- **Weight Tying**: `embed_tokens == lm_head`
- **GQA**: Reduced KV cache memory
- **KV Cache**: Inference acceleration
- **Mixed Precision**: FP16/BF16 training

### 3. Flexible Configuration

- Multiple model sizes (26M - 145M)
- Optional MoE architecture
- Optional Flash Attention
- LoRA fine-tuning support

## 📊 Model Performance

### Specification Comparison

| Model | Parameters | Memory | Training Time | Dialogue Capability |
|-------|------------|--------|---------------|---------------------|
| MiniMind2-Small | 26M | ~0.5GB | ~2h | Basic dialogue |
| MiniMind2-Base | 104M | ~1.5GB | ~6h | Fluent dialogue |
| MiniMind2-MoE | 145M | ~2GB | ~8h | High-quality dialogue |

### Use Cases

- **Learning & Research**: Deep understanding of LLM internal mechanisms
- **Rapid Prototyping**: Quick validation of algorithms and architectures
- **Edge Deployment**: Deployment in resource-constrained environments
- **Teaching Demo**: Complete LLM training pipeline demonstration

## 🔍 Comparison with Other Frameworks

| Feature | MiniMind | transformers+trl |
|---------|----------|------------------|
| Implementation | From scratch | Highly abstract |
| Learning Curve | Steep but deep | Gentle but shallow |
| Flexibility | Fully controllable | Limited by API |
| Use Case | Learning & research | Production |
| Code Volume | More | Very little |

## 📝 Summary

The core value of the MiniMind project lies in:

1. **Educational Significance**: Deep understanding of LLM core algorithms through from-scratch implementation
2. **Low Barrier**: Minimal cost and time, accessible to everyone
3. **Completeness**: Covers complete pipeline from pretraining to RLHF
4. **Extensibility**: Supports advanced features like MoE, multimodal, etc.

> "Building an airplane with LEGO is far more exciting than sitting in first class!"

## 📚 Related Resources

- **Vision-Language Version**: [MiniMind-V](https://github.com/jingyaogong/minimind-v)
- **Online Demo**: [ModelScope](https://www.modelscope.cn/studios/gongjy/MiniMind)
- **Video Introduction**: [Bilibili](https://www.bilibili.com/video/BV12dHPeqE72/)
- **Hugging Face Collection**: [Link](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

---

*Last Updated: 2026-03-24*
