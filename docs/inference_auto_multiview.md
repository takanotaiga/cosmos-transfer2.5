# Auto Multiview 推論ガイド

## 前提条件

1. 環境構築、チェックポイントのダウンロード、ハードウェア要件については [セットアップガイド](setup.md) に従ってください。

## 例
Multiview の実行には **8 GPU** が必要です。

multiview2world を実行:

```bash
export NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 -m examples.multiview --params_file assets/multiview_example/multiview_spec.json --num_gpus=$NUM_GPUS
```

## エンドツーエンドの Multiview 例

3D シーンアノテーションからマルチビュー動画出力までの一連のワークフローです。シーンアノテーション（物体位置、カメラキャリブレーション、車両軌跡）をレンダリングして、マルチビュー生成の条件となるワールドシナリオ動画を作成します。本例では、実写映像ではなくレンダリングされた制御動画のみを使用します。

**ステップ 1: シーンアノテーションをダウンロード**
```bash
mkdir -p datasets && curl -Lf https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/3d_scene_metadata.zip -o temp.zip && unzip temp.zip -d datasets && rm temp.zip
```

**ステップ 2: ワールドシナリオ動画を生成**
```bash
# See world_scenario_video_generation.md for detailed instructions
python scripts/generate_control_videos.py datasets/3d_scene_metadata assets/multiview_example1/world_scenario_videos
```

詳細な手順は [world_scenario_video_generation.md](world_scenario_video_generation.md) を参照してください。

**ステップ 3: マルチビュー推論を実行**
**重要な違い:** パラメータ JSON ファイルで { "num_conditional_frames": 0 } を設定します。
```bash

# 推論を実行（実写映像を使わないため num_conditional_frames=0）
export NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 -m examples.multiview --params_file assets/multiview_example/multiview_spec.json --num_gpus=$NUM_GPUS
```
