# Ultralytics 使用指南（机器人视觉项目版）

## 1. 安装

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO
```

## 2. 创建模型

```python
model = YOLO("yolo11n.pt")
```

模型规模：

- yolo11n.pt (Nano)
- yolo11s.pt (Small)
- yolo11m.pt (Medium)
- yolo11l.pt (Large)
- yolo11x.pt (XLarge)

## 3. 数据集结构

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

## 4. 标签格式

```text
class x_center y_center width height
```

所有坐标均归一化到[0,1]

## 5. data.yaml

```yaml
path: dataset

train: images/train
val: images/val

names:
  0: black_piece
  1: white_piece
```

## 6. 训练

命令行：

```bash
yolo detect train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640
```

Python：

```python
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

## 7. 验证

```python
metrics = model.val()
print(metrics.box.map50)
```

## 8. 推理

```python
model = YOLO("best.pt")

results = model("test.jpg")
```

## 9. 获取检测框

```python
for box in results[0].boxes:

    cls = int(box.cls)
    conf = float(box.conf)

    x1,y1,x2,y2 = box.xyxy[0]

    cx = (x1+x2)/2
    cy = (y1+y2)/2
```

## 10. 类别名称

```python
print(model.names)
```

## 11. 摄像头实时推理

```python
import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("result", annotated)

    if cv2.waitKey(1) == 27:
        break
```

## 12. 导出ONNX

```python
model.export(format="onnx")
```

## 13. 导出TensorRT

```python
model.export(format="engine")
```

## 14. 训练结果目录

```text
runs/
└── detect/
    └── train/
```

重要文件：

```text
best.pt
last.pt
results.png
confusion_matrix.png
```

## 15. 机器人井字棋项目流程

```text
RGB图
 ↓
YOLO检测黑棋/白棋
 ↓
bbox中心
 ↓
读取深度图
 ↓
计算三维坐标
 ↓
棋盘状态矩阵
 ↓
决策逻辑
 ↓
机械臂落子
```
