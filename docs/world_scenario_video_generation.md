# ワールドシナリオ動画の生成

Cosmos Transfer2 で使用するため、3D シーンアノテーションからワールドシナリオ動画を生成します。

## クイックスタート

```bash
# 依存関係をインストール
cd packages/cosmos-transfer2
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 制御動画を生成（既定では 7 カメラすべてを処理）
python scripts/generate_control_videos.py /path/to/{input_root} ./{save_root}
```

## 要件

- Python 3.10+
- UV（依存管理）
- EGL 対応 GPU（ヘッドレス OpenGL レンダリング用）
- Parquet 形式の 3D シーンアノテーションデータ

## 使い方

### 基本コマンド

```bash
# 全カメラ（既定）
python scripts/generate_control_videos.py {input_root}/ {save_root}/

# 特定のカメラのみ
python scripts/generate_control_videos.py {input_root}/ {save_root}/ \
    --cameras "camera:front:wide:120fov,camera:cross:right:120fov"
```

### オプション

| オプション | 既定 | 説明 |
|--------|---------|-------------|
| `--cameras` | `all` | カメラ名、または 7 カメラすべてを対象にする場合は "all" |

### 利用可能なカメラ

- `camera:front:wide:120fov`
- `camera:front:tele:sat:30fov`
- `camera:cross:right:120fov`
- `camera:cross:left:120fov`
- `camera:rear:left:70fov`
- `camera:rear:right:70fov`
- `camera:rear:tele:30fov`

## データ形式

### 入力構成
```
scene_annotations_directory/
├── uuid.obstacle.parquet              (required)
├── uuid.calibration_estimate.parquet  (required)
├── uuid.egomotion_estimate.parquet    (required)
├── uuid.lane.parquet                  (optional)
├── uuid.lane_line.parquet             (optional)
└── ... (other optional parquet files)
```

### 出力構成
```
save_root/
└── uuid/
    ├── uuid.camera_front_wide_120fov.mp4
    ├── uuid.camera_front_tele_sat_30fov.mp4
    ├── uuid.camera_cross_right_120fov.mp4
    ├── uuid.camera_cross_left_120fov.mp4
    ├── uuid.camera_rear_left_70fov.mp4
    ├── uuid.camera_rear_right_70fov.mp4
    ├── uuid.camera_rear_tele_30fov.mp4
```

## レンダリング内容

**常にレンダリング:** 車両/歩行者の 3D バウンディングボックス（必須の `obstacle.parquet` から）

**オプション（対応する parquet ファイルがある場合のみ）:**
- 車線、レーン、道路境界
- 横断歩道、ポール、路面標示、停止線
- 信号機、標識

## トラブルシューティング

### よくある問題

**ModernGL/EGL エラー**
→ GPU ドライバと EGL ライブラリ（`libGL.so.1`, `libEGL.so.1`）をインストール。Ubuntu/Debian の例: `apt install libegl1-mesa-dev libgl1-mesa-dri`

**parquet ファイル不足**
→ 必須ファイルが存在するか確認: obstacle, calibration_estimate, egomotion_estimate

**メモリ問題**
→ 同時に処理するカメラ数を減らす

**カメラ名が不正**
→ `--help` で有効なオプションを確認

## 連携

生成された制御動画は、Cosmos Transfer2 モデル推論の条件入力として使用できます。HD マップの可視化により、動画生成タスクに空間的コンテキストを提供します。
