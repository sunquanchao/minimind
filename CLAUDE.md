# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind is a minimalist Large Language Model (LLM) implementation from scratch using PyTorch. The project implements the complete LLM training pipeline including:
- **Pretraining**: Training from raw text data
- **SFT (Supervised Fine-Tuning)**: Full and LoRA-based fine-tuning
- **RLHF (Reinforcement Learning from Human Feedback)**: PPO, GRPO, DPO, and SPO algorithms
- **Reasoning training**: Specialized training for reasoning models (R1)
- **Model distillation**: Knowledge distillation from larger models
- **MoE (Mixture of Experts)**: Optional MoE architecture support

The smallest model (MiniMind2-Small) has only 26M parameters yet maintains conversational capabilities.

## Common Commands

### Model Inference
```bash
# Interactive chat with default settings
python eval_llm.py

# Manual input mode with specific model
python eval_llm.py --load_from model --weight full_sft --hidden_size 512 --num_hidden_layers 8 --historys 2

# Web UI demo (Streamlit)
streamlit run scripts/web_demo.py

# Start OpenAI-compatible API server
python scripts/serve_openai_api.py
```

### Training Pipeline
```bash
# 1. Train tokenizer (first time setup)
python -m trainer.train_tokenizer

# 2. Pretraining (from scratch)
python -m trainer.train_pretrain \
    --data_path ../dataset/pretrain_hq.jsonl \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 340 \
    --use_wandb

# 3. Supervised Fine-Tuning (full parameters)
python -m trainer.train_full_sft \
    --from_weight pretrain \
    --data_path ../dataset/sft_mini.jsonl \
    --epochs 2 \
    --batch_size 64 \
    --learning_rate 5e-5

# 4. LoRA Fine-Tuning (parameter-efficient)
python -m trainer.train_lora \
    --from_weight full_sft \
    --data_path ../dataset/sft_mini.jsonl \
    --lora_r 8 \
    --lora_alpha 32

# 5. DPO (Direct Preference Optimization)
python -m trainer.train_dpo \
    --from_weight full_sft \
    --data_path ../dataset/dpo_mini.jsonl

# 6. PPO (Proximal Policy Optimization)
python -m trainer.train_ppo \
    --from_weight full_sft \
    --reward_model_path ../reward_model

# 7. SPO (Self-Play Optimization)
python -m trainer.train_spo \
    --from_weight full_sft \
    --reasoning 1

# 8. Reasoning model training
python -m trainer.train_reason \
    --from_weight full_sft \
    --data_path ../dataset/reason_mini.jsonl
```

### Resume Training
Add `--from_resume 1` to any training script to automatically resume from the latest checkpoint.

### Distributed Training
```bash
# Single-node multi-GPU (example: 4 GPUs)
torchrun --nproc_per_node=4 -m trainer.train_pretrain --batch_size 8 ...
```

## Architecture

### Core Model Components (`model/model_minimind.py`)

- **MiniMindConfig**: Configuration class inheriting from `PretrainedConfig`
  - Supports both standard and MoE architectures
  - Configurable via `hidden_size`, `num_hidden_layers`, `num_attention_heads`
  - Model sizes: Small (26M), Base (104M), MoE (145M)

- **Attention**: Multi-head attention with GQA (Grouped Query Attention)
  - Rotary Position Embedding (RoPE) with YaRN scaling for long sequences
  - Optional Flash Attention support
  - KV cache for efficient inference

- **FeedForward**: SwiGLU activation function
  - Standard FFN for non-MoE models
  - MOEFeedForward with:
    - `n_routed_experts`: Total number of experts
    - `num_experts_per_tok`: Top-k experts selected per token
    - `n_shared_experts`: Shared experts for all tokens
    - Auxiliary loss for load balancing

- **MiniMindBlock**: Transformer decoder layer
  - Pre-LN architecture (RMSNorm)
  - Residual connections around attention and FFN

- **MiniMindForCausalLM**: Full causal LM model
  - Tied embeddings (embed_tokens = lm_head)
  - Supports generation with past_key_values

### Datasets (`dataset/lm_dataset.py`)

- **PretrainDataset**: For unsupervised pretraining from text
- **SFTDataset**: For instruction fine-tuning with conversation format
  - Handles multi-turn conversations
  - Generates labels only for assistant responses
  - Supports function calling via `tools` parameter
- **DPODataset**: For preference optimization (chosen/rejected pairs)
- **RLAIFDataset**: For RLHF training (prompts only)

### Training Utilities (`trainer/trainer_utils.py`)

- `init_model()`: Load model with optional weights
- `lm_checkpoint()`: Save/load training checkpoints with optimizer state
- `init_distributed_mode()`: Setup DDP training
- `SkipBatchSampler`: Resume training from specific step

## Training Workflow

The typical training progression is:
1. **Tokenizer**: Build vocabulary from training data
2. **Pretraining**: Learn language patterns from raw text (`train_pretrain.py`)
3. **SFT**: Align model to follow instructions (`train_full_sft.py`)
4. **Optional: LoRA**: Domain-specific fine-tuning (`train_lora.py`)
5. **Optional: RLHF**: Further alignment with human preferences
   - DPO for direct preference optimization (`train_dpo.py`)
   - PPO/GRPO/SPO for policy gradient methods
6. **Optional: Reasoning**: Train reasoning capabilities (`train_reason.py`)

### Key Training Arguments

Common to all trainers:
- `--save_dir`: Output directory for model weights
- `--save_weight`: Prefix for checkpoint files (e.g., 'pretrain', 'full_sft')
- `--from_weight`: Base checkpoint to load from
- `--from_resume`: Auto-detect and resume from checkpoint
- `--use_wandb`: Enable experiment tracking
- `--device`: CUDA device or CPU
- `--dtype`: Mixed precision (bfloat16 or float16)
- `--accumulation_steps`: Gradient accumulation for larger effective batch size
- `--use_compile`: Enable `torch.compile` for potential speedup

Model architecture:
- `--hidden_size`: Model dimension (512=Small, 768=Base)
- `--num_hidden_layers`: Number of transformer layers
- `--use_moe`: Enable Mixture of Experts

## Model Checkpoints

Checkpoints are saved to two locations:
1. `../out/{weight}_{hidden_size}.pth` - Model weights only
2. `../checkpoints/{weight}_{hidden_size}_resume.pth` - Full training state

Use `lm_checkpoint()` to save complete state for resumption.

## Notes

- All training scripts use DDP-aware logging (only main process prints)
- Models use weight tying (embed_tokens == lm_head) for efficiency
- MoE models append `_moe` suffix to checkpoint filenames
- Reasoning models use special `<think>` tags in responses
- The project implements core algorithms from scratch (no transformers+trl abstraction)
