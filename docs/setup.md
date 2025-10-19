# セットアップガイド

## システム要件

* Ampere 世代以降の NVIDIA GPU（RTX 30 シリーズ、A100 など）
* [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) に対応した NVIDIA ドライバ バージョン >= 570.124.06
* Linux x86-64
* glibc>=2.31（例: Ubuntu >= 22.04）
* Python 3.10

## インストール

リポジトリをクローンします:

```bash
git clone https://github.com/takanotaiga/cosmos-transfer2.5.git
cd cosmos-transfer2.5
```

システム依存関係をインストールします:

[uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

新しい仮想環境にパッケージをインストールします:

```shell
uv sync
source .venv/bin/activate
```

あるいは、現在アクティブな環境（例: conda）にインストールします:

```shell
uv sync --active --inexact
```

オプション依存関係も含めてインストールする場合:

```shell
uv sync --all-groups
```

## チェックポイントのダウンロード

1. `Read` 権限を持つ [Hugging Face アクセストークン](https://huggingface.co/settings/tokens) を取得
2. [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) をインストール: `uv tool install -U "huggingface_hub[cli]"`
3. ログイン: `hf auth login`
4. [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) に同意

推論およびポストトレーニングの実行中に、チェックポイントは自動的にダウンロードされます。キャッシュの保存先を変更する場合は、環境変数 [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) を設定してください。

## 高度な使用法

### Docker コンテナ

マシンで Docker が利用可能であり、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) がインストールされていることを確認してください。コンテナのビルド時にファイルディスクリプタ不足を避けるため、以下の例のように `--ulimit nofile` で上限を引き上げてください。毎回チェックポイントを再ダウンロードしないよう、環境変数 `HF_HOME` をコンテナからアクセス可能なパスに設定することを推奨します。最後に、並列実行の `torchrun` では共有メモリが不足しやすいため、`--shm-size` を大きくするか（セキュリティポリシーで許可されていれば）`--ipc=host` の使用を検討してください。

ビルド例:

```bash
docker build --ulimit nofile=131071:131071 -f Dockerfile . -t cosmos-transfer-2.5
```

実行例:

```bash
docker run --gpus all --rm -v .:/workspace -v /workspace/.venv -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface -it --ipc=host cosmos-transfer-2.5
```
