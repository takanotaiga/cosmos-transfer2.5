# ポストトレーニング ガイド

## 前提条件

### 1. 環境セットアップ

[セットアップガイド](./setup.md) に従い、依存関係のインストールを含む基本的な環境構築を行ってください。

### 2. Hugging Face の設定

モデルのチェックポイントが存在しない場合、ポストトレーニング中に自動でダウンロードされます。以下のように Hugging Face を設定してください:

```bash
# Hugging Face トークンでログイン（モデルのダウンロードに必須）
hf auth login

# HF モデルのカスタムキャッシュディレクトリを設定
# 既定: ~/.cache/huggingface
export HF_HOME=/path/to/your/hf/cache
```

> **💡 ヒント**: `HF_HOME` に十分なディスク容量があることを確認してください。

### 3. 学習出力ディレクトリ

学習のチェックポイントと成果物の保存先を設定します:

```bash
# 学習チェックポイント・成果物の出力先を設定
# 既定: /tmp/imaginaire4-output
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory
```

> **💡 ヒント**: `IMAGINAIRE_OUTPUT_ROOT` に十分なディスク容量があることを確認してください。

## Weights & Biases (W&B) へのロギング

既定では、学習は Weights & Biases にメトリクスを記録しようとします。以下の選択肢があります。

### オプション 1: W&B を有効化

W&B を用いた実験トラッキングを有効にするには:

1. [wandb.ai](https://wandb.ai) で無料アカウントを作成
2. [https://wandb.ai/authorize](https://wandb.ai/authorize) から API キーを取得
3. 以下の環境変数を設定:

    ```bash
    export WANDB_API_KEY=your_api_key_here
    ```

4. 次のコマンドで学習を開始します:

    ```bash
    EXP=your_experiment_name_here

    torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
      --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py  -- \
      experiment=${EXP}
    ```

### オプション 2: W&B を無効化

学習コマンドに `job.wandb_mode=disabled` を追加すると、W&B へのログ送信を無効化できます:

```bash
EXP=your_experiment_name_here

torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py -- \
  experiment=${EXP} \
  job.wandb_mode=disabled
```

## チェックポイント管理

本学習では 2 種類のチェックポイント形式を使用します。用途に応じて最適化されています。

### 1. 分散チェックポイント（DCP）形式

**学習時の主要チェックポイント形式** です。

- **構成**: モデル重みがシャーディングされた複数ファイルのディレクトリ
- **用途**: 学習中の保存、学習の再開
- **利点**:
  - マルチ GPU 学習における効率的な並列 I/O
  - FSDP（Fully Sharded Data Parallel）対応
  - 分散ワークロード向けに最適化

**ディレクトリ構成例:**

```
checkpoints/
├── iter_{NUMBER}/
│   ├── model/
│   │   ├── .metadata
│   │   └── __0_0.distcp
│   ├── optim/
│   ├── scheduler/
│   └── trainer/
└── latest_checkpoint.txt
```

### 2. 統合 PyTorch（.pt）形式

**推論や配布に適した単一ファイル形式** です。

- **構成**: モデルの完全な状態を含む単一の `.pt` ファイル
- **用途**: 推論、モデル共有、初回ポストトレーニングの開始
- **利点**:
  - 配布・バージョン管理が容易
  - 標準的な PyTorch 形式
  - 単一 GPU ワークフローで簡便

### チェックポイントの読み込み

学習システムは **両形式からの読み込みに対応** しています。

**DCP チェックポイントの読み込み（学習再開用）:**

```python
load_path="checkpoints/nvidia/Cosmos-Transfer2.5-2B/dcp"
```

**統合チェックポイントの読み込み（ポストトレーニング開始用）:**

```python
load_path="checkpoints/nvidia/Cosmos-Transfer2.5-2B/consolidated/model.pt"
```

> **注意**: Hugging Face からダウンロードした事前学習モデルは通常、統合 `.pt` 形式です。学習システムは自動的にこの形式を読み込み、学習を開始します。

### チェックポイントの保存

**学習中に保存されるチェックポイントはすべて DCP 形式です。** これにより次が保証されます:

- 学習実行間で一貫したチェックポイント構造
- 分散学習での最適な性能

## ポストトレーニングの例

詳細な学習例や設定オプションは以下を参照してください:

- [HDMap マルチビュー向け Control2World ポストトレーニング](./world_scenario_video_generation.md)
