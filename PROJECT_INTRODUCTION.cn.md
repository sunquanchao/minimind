# MiniMind 项目简介与架构介绍

## 📋 项目概述

**MiniMind** 是一个从零开始实现的大语言模型（LLM）训练框架，使用 PyTorch 原生代码重构所有核心算法。项目致力于降低 LLM 学习门槛，让开发者能够深入理解模型内部运作机制，而不仅仅是调用高度抽象的 API。

### 核心理念

- **从零实现**：所有核心算法（Attention、FFN、RoPE、MoE、PPO、DPO等）均使用 PyTorch 原生代码重构
- **极简设计**：最小模型仅 26M 参数，可在普通 GPU 上快速训练
- **全流程覆盖**：涵盖数据清洗、预训练、SFT、LoRA、RLHF、模型蒸馏等完整训练流程
- **低门槛入门**：成本极低（3元 + 2小时），适合学习研究

### 项目特色

| 特性 | 描述 |
|------|------|
| **超小体积** | MiniMind2-Small 仅 26M 参数，是 GPT-3 的 1/7000 |
| **快速训练** | 单卡 3090 约 2 小时完成训练 |
| **低成本** | GPU 服务器租用成本约 3 元 |
| **多模态支持** | 拓展版本 MiniMind-V 支持视觉多模态 |
| **全栈实现** | 不依赖 transformers+trl 的抽象接口 |

## 🏗️ 系统架构

### 整体架构图

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

### 目录结构

```
minimind/
├── model/                    # 模型架构定义
│   └── model_minimind.py    # MiniMind 模型核心实现
│       ├── MiniMindConfig   # 模型配置类
│       ├── Attention        # 多头注意力机制
│       ├── FeedForward      # 前馈神经网络
│       ├── MoEGate          # MoE 门控机制
│       └── MiniMindForCausalLM  # 完整因果语言模型
├── dataset/                  # 数据集处理
│   └── lm_dataset.py        # 数据集类定义
│       ├── PretrainDataset  # 预训练数据集
│       ├── SFTDataset       # 监督微调数据集
│       ├── DPODataset       # 偏好优化数据集
│       └── RLAIFDataset     # RLHF 训练数据集
├── trainer/                  # 训练脚本
│   ├── train_tokenizer.py   # 分词器训练
│   ├── train_pretrain.py    # 预训练
│   ├── train_full_sft.py    # 全量 SFT
│   ├── train_lora.py        # LoRA 微调
│   ├── train_dpo.py         # DPO 训练
│   ├── train_ppo.py         # PPO 训练
│   └── train_reason.py      # 推理模型训练
├── scripts/                  # 工具脚本
│   ├── web_demo.py          # Web UI 演示
│   └── serve_openai_api.py  # OpenAI 兼容 API
├── eval_llm.py              # 模型推理
└── requirements.txt         # 依赖列表
```

## 🔧 核心模块详解

### 1. 模型架构 (model/model_minimind.py)

#### 1.1 MiniMindConfig

模型配置类，继承自 `PretrainedConfig`，支持灵活的模型规模配置：

```python
# 基础配置
- hidden_size: 512                # 隐藏层维度
- num_hidden_layers: 8            # Transformer 层数
- num_attention_heads: 8          # 注意力头数
- num_key_value_heads: 2          # GQA KV 头数
- vocab_size: 6400                # 词汇表大小
- max_position_embeddings: 32768  # 最大序列长度

# MoE 配置（可选）
- use_moe: bool                   # 是否启用 MoE
- n_routed_experts: 4             # 路由专家总数
- num_experts_per_tok: 2          # 每个选择的专家数
- n_shared_experts: 1             # 共享专家数
```

#### 1.2 注意力机制 (Attention)

实现多头注意力机制，支持：
- **GQA (Grouped Query Attention)**：减少 KV cache 内存占用
- **RoPE (Rotary Position Embedding)**：旋转位置编码
- **YaRN 外推**：支持超长序列（最大 32K）
- **Flash Attention**：可选的高效注意力实现
- **KV Cache**：推理加速

```python
class Attention(nn.Module):
    # Q: 所有头
    # K/V: 仅 KV 头 (通过 repeat_kv 扩展)
    # RoPE: 旋转位置编码
    # Flash Attention: 可选加速
```

#### 1.3 前馈网络

**标准 FFN (FeedForward)**：
- SwiGLU 激活函数
- 门控机制：`act_fn(gate_proj(x)) * up_proj(x)`

**MoE FFN (MOEFeedForward)**：
```python
class MOEFeedForward(nn.Module):
    - MoEGate: 软路由门控机制
    - 专家选择：Top-K 路由
    - 负载均衡：辅助损失 (aux_loss)
    - 共享专家：所有 token 共享
```

#### 1.4 Transformer Block

采用 Pre-LN 架构：
```
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

#### 1.5 模型规格

| 模型 | 参数量 | hidden_size | layers | heads | 用途 |
|------|--------|-------------|--------|-------|------|
| MiniMind2-Small | 26M | 512 | 8 | 8/2 | 轻量级对话 |
| MiniMind2-Base | 104M | 768 | 12 | 12/4 | 标准对话 |
| MiniMind2-MoE | 145M | 768 | 12 | 12/4 | 高性能推理 |

### 2. 数据集处理 (dataset/lm_dataset.py)

#### 2.1 PretrainDataset

预训练数据集，从原始文本学习语言模式：
- 输入：JSON 格式的 `{text: "..."}`
- 处理：添加 BOS/EOS token，padding 到 max_length
- 标签：所有 token 都参与训练

#### 2.2 SFTDataset

监督微调数据集，用于指令遵循训练：
- 输入：多轮对话格式 `[{role, content}, ...]`
- 处理：
  - 支持多轮对话
  - 自动添加 system prompt（20% 概率）
  - 支持函数调用 (tools)
- 标签：**仅 assistant 回复部分**参与训练

#### 2.3 DPODataset

直接偏好优化数据集：
- 输入：chosen 和 rejected 对话对
- 输出：prompt + chosen/rejected response
- 损失掩码：仅在 assistant 回复计算损失

#### 2.4 RLAIFDataset

RLHF 训练数据集（PPO/GRPO）：
- 输入：prompt 列表
- 输出：用于强化学习生成的 prompt

### 3. 训练流程 (trainer/)

#### 3.1 训练阶段

```
1. Tokenizer 训练
   ↓
2. Pretraining (预训练)
   从随机初始化开始，学习语言基础能力
   ↓
3. SFT (监督微调)
   对齐模型理解指令，学习对话能力
   ↓
4. Optional: LoRA 微调
   高效参数微调，适配特定领域
   ↓
5. Optional: RLHF
   - DPO: 直接偏好优化
   - PPO/GRPO/SPO: 策略梯度优化
   ↓
6. Optional: Reasoning
   训练推理能力 (使用 <think> 标签)
```

#### 3.2 训练配置

```python
# 分布式训练
- DDP (DistributedDataParallel) 支持
- 梯度累积 (accumulation_steps)
- 混合精度 (bfloat16/float16)
- torch.compile 加速（可选）

# 检查点保存
- 模型权重：../out/{weight}_{hidden_size}.pth
- 完整状态：../checkpoints/{weight}_{hidden_size}_resume.pth
- 自动恢复：--from_resume 1

# 实验追踪
- WandB 集成：--use_wandb
```

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 训练 Tokenizer

```bash
python -m trainer.train_tokenizer
```

### 3. 预训练

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

### 4. 监督微调

```bash
python -m trainer.train_full_sft \
    --from_weight pretrain \
    --data_path ../dataset/sft_mini.jsonl \
    --epochs 2 \
    --batch_size 64 \
    --learning_rate 5e-5
```

### 5. 模型推理

```bash
# 交互式对话
python eval_llm.py

# Web UI
streamlit run scripts/web_demo.py

# API 服务
python scripts/serve_openai_api.py
```

## 💡 技术亮点

### 1. 从零实现的核心算法

- **Attention**：手动实现 QKV 投影、RoPE、Flash Attention
- **FFN/MoE**：原生实现 SwiGLU、专家路由、负载均衡
- **RoPE + YaRN**：手动实现位置编码和外推
- **RLHF**：手动实现 PPO、GRPO、DPO、SPO 算法

### 2. 高效设计

- **权重共享**：`embed_tokens == lm_head`
- **GQA**：减少 KV cache 内存
- **KV Cache**：推理加速
- **混合精度**：FP16/BF16 训练

### 3. 灵活配置

- 支持多种模型规模（26M - 145M）
- 可选 MoE 架构
- 可选 Flash Attention
- 支持 LoRA 微调

## 📊 模型性能

### 规格对比

| 模型 | 参数量 | 显存占用 | 训练时间 | 对话能力 |
|------|--------|----------|----------|----------|
| MiniMind2-Small | 26M | ~0.5GB | ~2h | 基础对话 |
| MiniMind2-Base | 104M | ~1.5GB | ~6h | 流畅对话 |
| MiniMind2-MoE | 145M | ~2GB | ~8h | 高质量对话 |

### 适用场景

- **学习研究**：深入理解 LLM 内部机制
- **快速验证**：算法和架构的快速原型
- **边缘部署**：资源受限环境部署
- **教学演示**：LLM 训练全流程教学

## 🔍 与其他框架对比

| 特性 | MiniMind | transformers+trl |
|------|----------|------------------|
| 实现方式 | 从零实现 | 高度抽象 |
| 学习曲线 | 陡峭但深入 | 平缓但浅层 |
| 灵活性 | 完全可控 | 受限于 API |
| 适用场景 | 学习研究 | 生产应用 |
| 代码量 | 较多 | 很少 |

## 📝 总结

MiniMind 项目的核心价值在于：

1. **教育意义**：通过从零实现，深入理解 LLM 核心算法
2. **低门槛**：极低的成本和时间，人人可参与
3. **完整性**：覆盖从预训练到 RLHF 的完整流程
4. **可扩展**：支持 MoE、多模态等高级特性

> "用乐高拼出一架飞机，远比坐在头等舱里飞行更让人兴奋！"

## 📚 相关资源

- **视觉多模态版本**：[MiniMind-V](https://github.com/jingyaogong/minimind-v)
- **在线体验**：[ModelScope](https://www.modelscope.cn/studios/gongjy/MiniMind)
- **视频介绍**：[Bilibili](https://www.bilibili.com/video/BV12dHPeqE72/)
- **Hugging Face Collection**：[链接](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

---

*最后更新：2026-03-24*
