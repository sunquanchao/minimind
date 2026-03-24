# MiniMind プロジェクト概要とアーキテクチャ紹介

## 📋 プロジェクト概要

**MiniMind** は、PyTorch のネイティブコードを使用してすべてのコアアルゴリズムをゼロから実装した大規模言語モデル（LLM）トレーニングフレームワークです。このプロジェクトは、LLM 学習の障壁を下げ、開発者が高度に抽象化された API を単に呼び出すだけでなく、モデルの内部メカニズムを深く理解できるようにすることを目指しています。

### コアコンセプト

- **ゼロからの実装**：すべてのコアアルゴリズム（Attention、FFN、RoPE、MoE、PPO、DPO など）を PyTorch のネイティブコードで再実装
- **ミニマリストデザイン**：最小のモデルはわずか 26M パラメータで、コンシューマー GPU で素早くトレーニング可能
- **完全なパイプラインカバレッジ**：データクリーニング、事前トレーニング、SFT、LoRA、RLHF、モデル蒸留まで完全なトレーニングワークフローをカバー
- **低エントリーバリア**：極めて低コスト（約3元 + 2時間）、学習や研究に最適

### プロジェクトの特徴

| 特徴 | 説明 |
|------|------|
| **超小型** | MiniMind2-Small はわずか 26M パラメータ、GPT-3 の 1/7000 |
| **高速トレーニング** | RTX 3090 1枚で約2時間 |
| **低コスト** | GPU サーバーレンタルコスト約3元 |
| **マルチモーダル対応** | 拡張版 MiniMind-V はビジョン言語タスクをサポート |
| **フルスタック実装** | transformers+trl の抽象インターフェースに依存しない |

## 🏗️ システムアーキテクチャ

### 全体アーキテクチャ図

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

### ディレクトリ構造

```
minimind/
├── model/                    # モデルアーキテクチャ定義
│   └── model_minimind.py    # MiniMind モデルコア実装
│       ├── MiniMindConfig   # モデル設定クラス
│       ├── Attention        # マルチヘッドアテンション機構
│       ├── FeedForward      # フィードフォワードニューラルネットワーク
│       ├── MoEGate          # MoE ゲート機構
│       └── MiniMindForCausalLM  # 完全な因果言語モデル
├── dataset/                  # データセット処理
│   └── lm_dataset.py        # データセットクラス定義
│       ├── PretrainDataset  # 事前トレーニングデータセット
│       ├── SFTDataset       # 教師付き微調整データセット
│       ├── DPODataset       # 好み最適化データセット
│       └── RLAIFDataset     # RLHF トレーニングデータセット
├── trainer/                  # トレーニングスクリプト
│   ├── train_tokenizer.py   # トークナイザー トレーニング
│   ├── train_pretrain.py    # 事前トレーニング
│   ├── train_full_sft.py    # 完全 SFT
│   ├── train_lora.py        # LoRA 微調整
│   ├── train_dpo.py         # DPO トレーニング
│   ├── train_ppo.py         # PPO トレーニング
│   └── train_reason.py      # 推論モデルトレーニング
├── scripts/                  # ユーティリティスクリプト
│   ├── web_demo.py          # Web UI デモ
│   └── serve_openai_api.py  # OpenAI 互換 API
├── eval_llm.py              # モデル推論
└── requirements.txt         # 依存関係リスト
```

## 🔧 コアモジュール詳細

### 1. モデルアーキテクチャ (model/model_minimind.py)

#### 1.1 MiniMindConfig

`PretrainedConfig` を継承したモデル設定クラス、柔軟なモデルスケーリングをサポート：

```python
# 基本設定
- hidden_size: 512                # 隠れ層の次元
- num_hidden_layers: 8            # Transformer 層の数
- num_attention_heads: 8          # アテンションヘッドの数
- num_key_value_heads: 2          # GQA KV ヘッドの数
- vocab_size: 6400                # 語彙サイズ
- max_position_embeddings: 32768  # 最大シーケンス長

# MoE 設定（オプション）
- use_moe: bool                   # MoE を有効にするかどうか
- n_routed_experts: 4             # ルーティング専門家の総数
- num_experts_per_tok: 2          # トークンごとに選択される専門家の数
- n_shared_experts: 1             # 共有専門家の数
```

#### 1.2 アテンション機構 (Attention)

マルチヘッドアテンション機構を実装、以下をサポート：
- **GQA (Grouped Query Attention)**：KV cache メモリ使用量を削減
- **RoPE (Rotary Position Embedding)**：回転位置エンコーディング
- **YaRN 外挿**：超長シーケンスをサポート（最大 32K）
- **Flash Attention**：オプションの効率的なアテンション実装
- **KV Cache**：推論加速

```python
class Attention(nn.Module):
    # Q: すべてのヘッド
    # K/V: KV ヘッドのみ（repeat_kv で拡張）
    # RoPE: 回転位置エンコーディング
    # Flash Attention: オプションの加速
```

#### 1.3 フィードフォワードネットワーク

**標準 FFN (FeedForward)**：
- SwiGLU 活性化関数
- ゲート機構：`act_fn(gate_proj(x)) * up_proj(x)`

**MoE FFN (MOEFeedForward)**：
```python
class MOEFeedForward(nn.Module):
    - MoEGate: ソフトルーティングゲート機構
    - 専門家選択：Top-K ルーティング
    - 負荷分散：補助損失 (aux_loss)
    - 共有専門家：すべてのトークンで共有
```

#### 1.4 Transformer ブロック

Pre-LN アーキテクチャを採用：
```
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

#### 1.5 モデル仕様

| モデル | パラメータ数 | hidden_size | 層数 | ヘッド数 | 用途 |
|-------|-------------|-------------|------|---------|------|
| MiniMind2-Small | 26M | 512 | 8 | 8/2 | 軽量対話 |
| MiniMind2-Base | 104M | 768 | 12 | 12/4 | 標準対話 |
| MiniMind2-MoE | 145M | 768 | 12 | 12/4 | 高性能推論 |

### 2. データセット処理 (dataset/lm_dataset.py)

#### 2.1 PretrainDataset

生テキストから言語パターンを学習するための事前トレーニングデータセット：
- 入力：JSON 形式 `{text: "..."}`
- 処理：BOS/EOS トークンを追加、max_length までパディング
- ラベル：すべてのトークンがトレーニングに参加

#### 2.2 SFTDataset

命令に従うための教師付き微調整データセット：
- 入力：マルチターンダイアログ形式 `[{role, content}, ...]`
- 処理：
  - マルチターンダイアログをサポート
  - 自動 system prompt 追加（20% の確率）
  - 関数呼び出しをサポート (tools)
- ラベル：**アシスタントの応答部分のみ**がトレーニングに参加

#### 2.3 DPODataset

直接的好み最適化データセット：
- 入力：chosen と rejected のダイアログペア
- 出力：prompt + chosen/rejected response
- 損失マスク：アシスタント応答でのみ損失を計算

#### 2.4 RLAIFDataset

RLHF トレーニングデータセット（PPO/GRPO）：
- 入力：プロンプトリスト
- 出力：強化学習生成用のプロンプト

### 3. トレーニングパイプライン (trainer/)

#### 3.1 トレーニング段階

```
1. トークナイザー トレーニング
   ↓
2. 事前トレーニング (Pretraining)
   ランダム初期化から基本言語能力を学習
   ↓
3. SFT (教師付き微調整)
   モデルを命令理解に合わせ、対話能力を学習
   ↓
4. オプション: LoRA 微調整
   効率的なパラメータ微調整、ドメイン適応
   ↓
5. オプション: RLHF
   - DPO: 直接的好み最適化
   - PPO/GRPO/SPO: 政策勾配最適化
   ↓
6. オプション: 推論 (Reasoning)
   推論能力をトレーニング（｜think＞タグ使用）
```

#### 3.2 トレーニング設定

```python
# 分散トレーニング
- DDP (DistributedDataParallel) サポート
- 勾配蓄積 (accumulation_steps)
- 混合精度 (bfloat16/float16)
- torch.compile 加速（オプション）

# チェックポイント保存
- モデル重み：../out/{weight}_{hidden_size}.pth
- 完全状態：../checkpoints/{weight}_{hidden_size}_resume.pth
- 自動復元：--from_resume 1

# 実験追跡
- WandB 統合：--use_wandb
```

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
pip install -r requirements.txt
```

### 2. トークナイザー トレーニング

```bash
python -m trainer.train_tokenizer
```

### 3. 事前トレーニング

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

### 4. 教師付き微調整

```bash
python -m trainer.train_full_sft \
    --from_weight pretrain \
    --data_path ../dataset/sft_mini.jsonl \
    --epochs 2 \
    --batch_size 64 \
    --learning_rate 5e-5
```

### 5. モデル推論

```bash
# インタラクティブ対話
python eval_llm.py

# Web UI
streamlit run scripts/web_demo.py

# API サービス
python scripts/serve_openai_api.py
```

## 💡 技術的ハイライト

### 1. ゼロから実装されたコアアルゴリズム

- **Attention**：QKV 投影、RoPE、Flash Attention の手動実装
- **FFN/MoE**：SwiGLU、専門家ルーティング、負荷分散のネイティブ実装
- **RoPE + YaRN**：位置エンコーディングと外挿の手動実装
- **RLHF**：PPO、GRPO、DPO、SPO アルゴリズムの手動実装

### 2. 効率的なデザイン

- **重み共有**：`embed_tokens == lm_head`
- **GQA**：KV cache メモリ削減
- **KV Cache**：推論加速
- **混合精度**：FP16/BF16 トレーニング

### 3. 柔軟な設定

- 複数のモデルサイズ（26M - 145M）
- オプションの MoE アーキテクチャ
- オプションの Flash Attention
- LoRA 微調整サポート

## 📊 モデル性能

### 仕様比較

| モデル | パラメータ数 | メモリ | トレーニング時間 | 対話能力 |
|-------|-------------|--------|------------------|----------|
| MiniMind2-Small | 26M | ~0.5GB | ~2h | 基本的な対話 |
| MiniMind2-Base | 104M | ~1.5GB | ~6h | 流暢な対話 |
| MiniMind2-MoE | 145M | ~2GB | ~8h | 高品質な対話 |

### 使用シーン

- **学習・研究**：LLM 内部メカニズムの深い理解
- **迅速なプロトタイピング**：アルゴリズムとアーキテクチャの迅速な検証
- **エッジ展開**：リソース制約のある環境への展開
- **教育デモ**：完全な LLM トレーニングパイプラインのデモンストレーション

## 🔍 他のフレームワークとの比較

| 特徴 | MiniMind | transformers+trl |
|------|----------|------------------|
| 実装方式 | ゼロから実装 | 高度に抽象化 |
| 学習曲線 | 急だが深い | 緩やかだが浅い |
| 柔軟性 | 完全に制御可能 | API に制限される |
| 使用シーン | 学習・研究 | 本番環境 |
| コード量 | 多い | 非常に少ない |

## 📝 まとめ

MiniMind プロジェクトの核心的な価値は：

1. **教育的意義**：ゼロからの実装を通じて LLM コアアルゴリズムを深く理解
2. **低い障壁**：最小限のコストと時間、誰でもアクセス可能
3. **完全性**：事前トレーニングから RLHF まで完全なパイプラインをカバー
4. **拡張性**：MoE、マルチモーダルなどの高度な機能をサポート

> 「レゴで飛行機を組み立てることは、ファーストクラスに座って飛ぶことよりもはるかにエキサイティングだ！」

## 📚 関連リソース

- **ビジョン言語版**：[MiniMind-V](https://github.com/jingyaogong/minimind-v)
- **オンラインデモ**：[ModelScope](https://www.modelscope.cn/studios/gongjy/MiniMind)
- **動画紹介**：[Bilibili](https://www.bilibili.com/video/BV12dHPeqE72/)
- **Hugging Face コレクション**：[リンク](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

---

*最終更新：2026-03-24*
