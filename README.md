# Tic-Tac-Toe Robot Perception and Control

This package implements a physical tic-tac-toe robot pipeline. It supports YOLO-based board/piece detection, YOLO+SAM mask-enhanced perception, optional Qwen2.5-VL board-state arbitration, minimax rule decision, and template-based robot pick-and-place execution.

The current codebase is organized for two main use cases:

- Offline perception experiments for paper metrics and visual reports.
- ROS-based physical robot control using the existing pick/place template actions.

## Project Structure

```text
tic_tac_toe/
|-- config/
|   |-- config.yaml          # Main runtime configuration
|   |-- data.yaml            # YOLO dataset config
|   `-- classes.txt          # Class names: board, yellow_piece, blue_piece
|-- prepare/
|   |-- collect_dataset.py   # Image collection utility
|   `-- split_dataset.py     # Train/val split utility
|-- scripts/
|   |-- main_control.py                  # YOLO + minimax + robot control
|   |-- yolo_sam_control.py              # YOLO+SAM + minimax + robot control
|   |-- yolo_sam_perception_frontend.py  # Offline YOLO+SAM(+VLM) perception
|   |-- yolo_sam_val_report.py           # Validation report and paper metrics
|   |-- execution_chain_experiment.py    # Physical execution-chain experiment
|   |-- tictactoe_engine.py              # Tic-tac-toe rules and minimax
|   |-- tictactoe_pick_place.py          # Template pick/place execution
|   `-- vision_state.py                  # YOLO board-state perception
|-- train_yolo.py            # YOLO training entry
|-- requirements.txt         # ROS/Linux dependencies
|-- requirements-win.txt     # Windows/offline perception dependencies
`-- requirements-vlm.txt     # VLM experiment dependencies
```

Large runtime files are intentionally not tracked by Git:

```text
dataset/
runs/
*.pt
*.pth
*.safetensors
__pycache__/
```

## Environment Setup

For Windows or non-ROS offline perception:

```bash
pip install -r requirements-win.txt
```

For Qwen2.5-VL experiments:

```bash
pip install -r requirements-vlm.txt
```

For ROS robot control on Linux, install the base requirements in the ROS Python environment:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `rospy` and `cv_bridge`, which are usually provided by the ROS environment rather than normal PyPI on Windows.

## Required Local Files

The following files are needed for normal experiments but are not committed to Git:

```text
FastSAM-s.pt
runs/detect/runs/tic_tac_toe_yolo/weights/best.pt
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

The YOLO class mapping is:

```text
0: board
1: yellow_piece
2: blue_piece
```

The board state is a length-9 list:

```text
0 = empty
1 = yellow_piece
2 = blue_piece
```

Cell indexes are ordered row-by-row:

```text
0 1 2
3 4 5
6 7 8
```

## Configuration

Main parameters are in:

```text
config/config.yaml
```

Important sections:

- `yolo`: YOLO model path, confidence threshold, device, class names.
- `yolo_sam`: FastSAM path, SAM thresholds, mask matching, fallback behavior, color checks.
- `vlm`: Qwen2.5-VL model, device map, max generated tokens, quantization mode.
- `vision`: board grid ratios and display settings.
- `game`: tic-tac-toe piece values, minimax preferences, stability checks.
- `arm_pick_place`: robot template actions and per-cell target keyframes.

VLM quantization can be controlled from config:

```yaml
vlm:
  enabled: true
  model: Qwen/Qwen2.5-VL-7B-Instruct
  device_map: auto
  max_new_tokens: 512
  quantization: none   # none, 8bit, or 4bit
```

## Dataset Collection

Collect images from a USB camera:

```bash
python prepare/collect_dataset.py --source 0 --out dataset/images/train
```

Collect images from a ROS image topic:

```bash
python prepare/collect_dataset.py --ros --source /camera/color/image_raw --out dataset/images/train
```

Preview keys:

- `s`: save one frame
- `a`: toggle auto-save
- `q` or `esc`: quit

After collecting images, label them in YOLO format using the classes in `config/classes.txt`.

Split the dataset:

```bash
python prepare/split_dataset.py --root dataset --val-ratio 0.2
```

Use `--copy` if validation files should be copied instead of moved:

```bash
python prepare/split_dataset.py --root dataset --val-ratio 0.2 --copy
```

## YOLO Training

Train the detector:

```bash
python train_yolo.py --data config/data.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

Common options:

```bash
python train_yolo.py \
  --data config/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --project runs/detect/runs \
  --name tic_tac_toe_yolo
```

The default trained model path used by `config/config.yaml` is:

```text
runs/detect/runs/tic_tac_toe_yolo/weights/best.pt
```

## Offline YOLO+SAM Perception

Run one image:

```bash
python scripts/yolo_sam_perception_frontend.py --source dataset/images/val/000009.jpg
```

Run a directory:

```bash
python scripts/yolo_sam_perception_frontend.py --source dataset/images/val --max-images 20
```

Enable Qwen2.5-VL arbitration:

```bash
python scripts/yolo_sam_perception_frontend.py \
  --source dataset/images/val/000009.jpg \
  --enable-vlm
```

Use 4-bit VLM quantization:

```bash
python scripts/yolo_sam_perception_frontend.py \
  --source dataset/images/val/000009.jpg \
  --enable-vlm \
  --vlm-quant 4bit \
  --vlm-max-new-tokens 128
```

Outputs are written to:

```text
runs/yolo_sam_frontend/
```

Typical outputs include JSON, annotated image, per-piece masks, and a combined all-mask preview image.

## Validation Report for Paper Metrics

Run the full validation report with VLM enabled:

```bash
python scripts/yolo_sam_val_report.py
```

Run without VLM:

```bash
python scripts/yolo_sam_val_report.py --no-vlm
```

Run a quick subset:

```bash
python scripts/yolo_sam_val_report.py --max-images 5
```

Compare quantization settings:

```bash
python scripts/yolo_sam_val_report.py --vlm-quant none --out runs/yolo_sam_frontend/val_report_none
python scripts/yolo_sam_val_report.py --vlm-quant 8bit --out runs/yolo_sam_frontend/val_report_8bit
python scripts/yolo_sam_val_report.py --vlm-quant 4bit --out runs/yolo_sam_frontend/val_report_4bit
```

The report saves:

```text
runs/yolo_sam_frontend/val_report/
├── summary.json
├── summary.csv
├── metrics_summary.jpg
├── confusion_yolo_only.jpg
├── confusion_yolo_sam_raw.jpg
├── confusion_vlm_raw.jpg
├── confusion_yolo_sam_vlm_final.jpg
└── images/
```

Main paper metrics include:

- board-state accuracy
- cell classification accuracy
- grid assignment accuracy
- illegal state interception rate
- final legal state rate
- low-confidence sample recognition rate
- average inference time
- mask fallback rate

`summary.json` also records the VLM experiment setting:

```json
"vlm_quantization": "4bit",
"vlm_config": {
  "enabled": true,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "device_map": "auto",
  "max_new_tokens": 128,
  "quantization": "4bit"
}
```

## ROS Robot Control

Dry-run one YOLO-based move without moving the arm:

```bash
python scripts/main_control.py --once --dry-run
```

Run YOLO + minimax + template pick/place:

```bash
python scripts/main_control.py
```

Run YOLO+SAM + minimax + template pick/place:

```bash
python scripts/yolo_sam_control.py
```

Run only one physical move:

```bash
python scripts/yolo_sam_control.py --once
```

Important ROS/config assumptions:

- Camera topic: `/camera/color/image_raw`
- Board state topic: `/tic_tac_toe/board_state`
- Next move topic: `/tic_tac_toe/next_move`
- Arm mode service: `/arm_traj_change_mode`
- Arm joint target topic: `/kuavo_arm_target_poses`
- Pick/place execution is currently template-based through `TicTacToePickPlace.place_piece_to_cell(target_cell)`.

Before physical execution, verify all target keyframes under:

```yaml
arm_pick_place:
  put_piece_targets:
```

## Execution-Chain Experiment

Use this script to collect physical execution metrics required by the paper.

Dry-run data collection without robot motion:

```bash
python scripts/execution_chain_experiment.py \
  --target-cells 0,1,2,3,4,5,6,7,8 \
  --repeats 3 \
  --non-interactive
```

Execute real robot template actions:

```bash
python scripts/execution_chain_experiment.py \
  --target-cells 0,1,2,3,4,5,6,7,8 \
  --repeats 3 \
  --execute
```

Record a strategy label:

```bash
python scripts/execution_chain_experiment.py \
  --strategy template_param \
  --strategy-params "{\"height_offset\": 0.01}" \
  --target-cells 0,1,2 \
  --repeats 5 \
  --execute
```

Supported strategy labels:

```text
template
template_param
template_residual
template_residual_rl
```

At present, real execution still uses the template backend unless additional residual or policy code is connected. The strategy label is recorded for experiment comparison.

Outputs are saved under:

```text
runs/execution_chain_experiment/
```

Main execution metrics include:

- placement success rate
- target-cell hit rate
- post-execution state consistency rate
- failure recovery success rate
- average closed-loop time

## Git Workflow

The repository is intended to save code only. Dataset, model weights, and experiment outputs stay local.

Typical development workflow:

```bash
git checkout dev
git pull
```

After code changes:

```bash
git status
git add .
git commit -m "Describe the change"
git push
```

On the training machine, if local experiment outputs should be preserved while code is overwritten from GitHub:

```bash
git fetch origin
git reset --hard origin/dev
```

This only overwrites tracked code files. Ignored local files such as `dataset/`, `runs/`, and `*.pt` remain untouched.

## Notes

- Use `--dry-run` before sending commands to the real robot.
- VLM inference is much slower than YOLO/SAM and may require a GPU with enough VRAM.
- 4-bit or 8-bit VLM quantization reduces memory pressure and is useful for local deployment experiments.
- If CUDA is unavailable, the perception scripts fall back to CPU where supported, but Qwen2.5-VL inference may become very slow.
