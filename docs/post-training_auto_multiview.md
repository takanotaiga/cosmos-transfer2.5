# HDMap 向け Auto Multiview ポストトレーニング

このガイドでは、Cosmos-Transfer2.5 Auto Multiview 2B モデルでポストトレーニングを実行する手順を説明します。

## 目次

- [前提条件](#prerequisites)
- [データ準備](#1-preparing-data)
- [ポストトレーニング](#2-post-training)
- [ポストトレーニング後のチェックポイントで推論](#3-inference-with-the-post-trained-checkpoint)

<a id="prerequisites"></a>
## 前提条件

先へ進む前に、チェックポイントの扱いやベストプラクティスを含む詳細な手順について [ポストトレーニング ガイド](./post-training.md) を必ずお読みください。Cosmos-Transfer2.5 でのポストトレーニングを円滑に行う準備が整います。

<a id="1-preparing-data"></a>
## 1. データ準備

### 1.1 Transfer マルチビュー学習データセットの用意

最初のステップは、動画を用意したデータセットの準備です。

各サンプルに対して、できれば 720p の **MP4 形式**の動画群を含むフォルダと、対応する **MP4 形式**の HDMap 制御入力動画群を含むフォルダを用意してください。各サンプルのビューは、カメラ名のサブディレクトリに分けて配置します。`assets/multiview_hdmap_posttrain_dataset` に使用可能なサンプルデータセットがあります。

### 1.2 データセットフォルダ形式の確認

データセットのフォルダ構成例:

```
assets/multiview_hdmap_posttrain_dataset/
├── captions/
│   └── ftheta_camera_front_wide_120fov/
│       └── *.json
├── control_input_hdmap_bbox/
│   ├── ftheta_camera_cross_left_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_cross_right_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_wide_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_tele_30fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_left_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_right_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_tele_30fov/
│   │   └── *.mp4
├── videos/
│   ├── ftheta_camera_cross_left_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_cross_right_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_wide_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_tele_30fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_left_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_right_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_tele_30fov/
│   │   └── *.mp4
```

<a id="2-post-training"></a>
## 2. ポストトレーニング

以下のコマンドで、マルチビューデータを用いたポストトレーニングの例を実行します。

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=transfer2_auto_multiview_post_train_example
```

モデルはマルチビューのデータセットでポストトレーニングされます。データローダーの定義は [データ設定](../projects/cosmos/transfer2_multiview/configs/vid2vid_transfer/defaults/data.py) を参照してください。

チェックポイントは `${IMAGINAIRE_OUTPUT_ROOT}/PROJECT/GROUP/NAME/checkpoints` に保存されます。既定の `IMAGINAIRE_OUTPUT_ROOT` は `/tmp/imaginaire4-output` です。チェックポイントのために十分なストレージ容量のある場所に `IMAGINAIRE_OUTPUT_ROOT` を設定することを強く推奨します。

上記の例では、`PROJECT` は `cosmos_transfer_v2p5`、`GROUP` は `auto_multiview`、`NAME` は `2b_cosmos_multiview_post_train_example` です。

それらがどのように決定されるかは、ジョブ設定を参照してください。

```python
transfer2_auto_multiview_post_train_example = dict(
    dict(
        ...
        job=dict(
            project="cosmos_transfer_v2p5",
            group="auto_multiview",
            name="2b_cosmos_multiview_post_train_example"
        ),
        ...
    )
)
```

<a id="3-inference-with-the-post-trained-checkpoint"></a>
## 3. ポストトレーニング後のチェックポイントで推論

### 3.1 DCP チェックポイントを統合 PyTorch 形式に変換

学習中に保存されるチェックポイントは DCP 形式のため、推論では統合 PyTorch 形式（.pt）へ変換する必要があります。`convert_distcp_to_pt.py` スクリプトを使用します:

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_transfer_v2p5/auto_multiview/2b_cosmos_multiview_post_train_example/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

変換により以下の 3 ファイルが生成されます:

- `model.pt`: 通常重み + EMA 重みを含むフルチェックポイント
- `model_ema_fp32.pt`: EMA 重み（float32）
- `model_ema_bf16.pt`: EMA 重み（bfloat16、推論に推奨）

### 3.2 推論の実行

チェックポイントの変換後、推論パラメータを指定した JSON 設定ファイルを用いて、ポストトレーニング済みモデルで推論を実行できます（例: `assets/multiview_example/multiview_spec.json`）。

```bash
export NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 -m examples.multiview --params_file assets/multiview_example/multiview_spec.json --num_gpus=$NUM_GPUS --checkpoint_path $CHECKPOINT_DIR/model_ema_bf16.pt --experiment transfer2_auto_multiview_post_train_example
```

生成された動画は出力ディレクトリ（例: `outputs/multiview_control2world/`）に保存されます。

推論オプションや高度な使い方の詳細は [docs/inference.md](./inference.md) を参照してください。
