# Tic-Tac-Toe Vision Workflow

## 1. Collect YOLO images

USB camera or video:

```bash
python3 collect_dataset.py --source 0 --out dataset/images/train
```

ROS image topic:

```bash
python3 collect_dataset.py --ros --source /camera/color/image_raw --out dataset/images/train
```

Keys in preview window:

- `s`: save one frame
- `a`: toggle auto-save
- `q` or `esc`: quit

After collection, label images in YOLO format. The default classes are:

- `0`: `black_piece`
- `1`: `white_piece`

## 2. Split train/val

```bash
python split_dataset.py --root dataset --val-ratio 0.2
```

Use `--copy` if you want to copy validation files instead of moving them.

## 3. Train YOLO

```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16
```
yolo detect train \
  model=yolo11n.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=8 \
  project=runs \
  name=tic_tac_toe_yolo \
  amp=False
The default output is:

```text
runs/detect/tic_tac_toe_yolo/weights/best.pt
```

## 4. YOLO inference

If the board fills the image:

```bash
python infer.py --weights runs/detect/tic_tac_toe_yolo/weights/best.pt --source test.jpg --save-json
```

If you already know the board box:

```bash
python infer.py --weights runs/detect/tic_tac_toe_yolo/weights/best.pt --source test.jpg --board 120 80 520 480 --save-json
```

Output includes:

- annotated image in `runs/infer/yolo`
- optional JSON with `board_box`, `state`, and detected `pieces`

The `state` matrix uses:

- `X`: black piece
- `O`: white piece
- `-`: empty

## 5. FastSAM inference

Auto-detect board:

```bash
python infer_sam.py --model FastSAM-s.pt --source test.jpg --save-json
```

Known board box:

```bash
python infer_sam.py --model FastSAM-s.pt --source test.jpg --board 120 80 520 480 --save-json
```

Tune color thresholds when lighting changes:

```bash
python infer_sam.py --source test.jpg --black-threshold 90 --white-threshold 175 --save-json
```

Output includes annotated images and optional JSON in `runs/infer/fastsam`.

## 6. Real robot control

Before running on Roban2, calibrate these values in `config/config.yaml`:

- `frames.geometry_frame`, `frames.ik_frame`, `frames.geometry_to_ik_translation_xyz`
- `board.board_center`, `board.cell_size`, `board.z_height`
- `piece.pick_z_height`, `piece.place_z_height`, `piece.safe_z_offset`
- `motion.right_quat_xyzw`, `motion.right_tool_offset_xyz`
- `hand.interface`, `hand.open_position`, `hand.close_position`

By default the board and spare-piece coordinates are measured in `odom`, while
the IK request is sent in `base_link`. With `base_link` 0.73 m above `odom`,
the configured transform is `[0.0, 0.0, -0.73]`.

Dry-run one move without moving the arm:

```bash
python3 scripts/main_control.py --once --dry-run
```

Run the YOLO + minimax + Roban2 arm loop:

```bash
python3 scripts/main_control.py
```

Run the YOLO + SAM vision fusion with the same minimax + Roban2 arm loop:

```bash
python3 scripts/yolo_sam_control.py
```

The controller uses:

- `/camera/color/image_raw` for YOLO board state
- `/arm_traj_change_mode` to switch the arm to external control mode `2`
- `/ik/two_arm_hand_pose_cmd_srv` for right-hand Cartesian IK
- `/kuavo_arm_target_poses` to publish the 14 arm joint targets
- `/dexhand/right/command` by default for right-hand gripper control
