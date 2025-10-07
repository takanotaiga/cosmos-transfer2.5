# World Scenario Video Generation

Generate world scenario videos from 3D scene annotations for use with Cosmos Transfer2.

## Quick Start

```bash
# Install dependencies
cd packages/cosmos-transfer2
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Generate control videos (processes all 7 cameras by default)
python scripts/generate_control_videos.py /path/to/{input_root} ./{save_root}
```

## Requirements

- Python 3.10+
- UV (for dependency management)
- GPU with EGL support (for headless OpenGL rendering)
- 3D scene annotation data in parquet format

## Usage

### Basic Commands

```bash
# All cameras (default)
python scripts/generate_control_videos.py {input_root}/ {save_root}/

# Specific cameras
python scripts/generate_control_videos.py {input_root}/ {save_root}/ \
    --cameras "camera:front:wide:120fov,camera:cross:right:120fov"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cameras` | `all` | Camera names or "all" for all 7 cameras |

### Available Cameras

- `camera:front:wide:120fov`
- `camera:front:tele:sat:30fov`
- `camera:cross:right:120fov`
- `camera:cross:left:120fov`
- `camera:rear:left:70fov`
- `camera:rear:right:70fov`
- `camera:rear:tele:30fov`

## Data Format

### Input Structure
```
scene_annotations_directory/
├── uuid.obstacle.parquet              (required)
├── uuid.calibration_estimate.parquet  (required)
├── uuid.egomotion_estimate.parquet    (required)
├── uuid.lane.parquet                  (optional)
├── uuid.lane_line.parquet             (optional)
└── ... (other optional parquet files)
```

### Output Structure
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

## What Gets Rendered

**Always rendered:** 3D bounding boxes for vehicles/pedestrians (from required `obstacle.parquet`)

**Optionally rendered** (only if corresponding parquet file provided):
- Lane lines, lanes, road boundaries
- Crosswalks, poles, road markings, wait lines
- Traffic lights and signs

## Troubleshooting

### Common Issues

**ModernGL/EGL errors**
→ Install GPU drivers and EGL libraries (`libGL.so.1`, `libEGL.so.1`). On Ubuntu/Debian: `apt install libegl1-mesa-dev libgl1-mesa-dri`

**Missing parquet files**
→ Ensure required files exist: obstacle, calibration_estimate, egomotion_estimate

**Memory issues**
→ Process fewer cameras at once if needed

**Invalid camera names**
→ Run with `--help` to see valid options

## Integration

Generated control videos serve as conditioning inputs for Cosmos Transfer2 model inference. The HD map visualizations provide spatial context for video generation tasks.
