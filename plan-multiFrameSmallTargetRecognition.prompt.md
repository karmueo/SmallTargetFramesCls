# Plan: 多帧小目标识别系统

基于 2D CNN 提取空间特征与位置/运动信息，利用 TCN 进行时序建模，实现鸟、无人机、夜间无人机、飞机等小目标的多帧序列分类。

## Steps

1. 数据管线：`src/data/dataset.py` 支持序列加载、类别枚举映射（bird/drone/night_drone/plane 等）、CSV 列表读取（train/val/test）与数据一致性校验（帧-JSON 对齐、bbox 合法性）。
2. 数据增强与采样：`src/data/transforms.py` 定义小幅度仿射/光照/模糊/噪声增强；帧采样支持均匀/随机起点-等间隔、长度不足的重复/填充策略。
3. 视觉特征：`src/models/feature_extractor.py` 采用轻量级 CNN（如 MobileNetV3-small 自定义变体），兼容单通道/三通道输入，可选预训练开关。
4. 时序分类器：`src/models/classifier.py` 融合 CNN 特征与 TCN，明确 kernel size、dilation、残差/LayerNorm、dropout 配置。
5. 训练与配置：`configs/default.yaml` 定义 batch size、采样帧数、优化器与调度（如 Cosine/Step）、warmup、梯度裁剪、混合精度；`scripts/train.py` 完成训练循环与检查点。
6. 融合位置/运动特征：`src/models/position_encoder.py` 实现位置（归一化坐标）、速度、加速度等编码方式，融合到时序输入中提升性能。
7. 评估与可视化：`scripts/eval.py` 输出 per-class F1、宏 F1、混淆矩阵；`src/utils/visualization.py` 支持叠加轨迹与预测；日志可选 TensorBoard/W&B。
8. 推理与导出：`scripts/infer.py` 支持单序列/滑窗推理，输出 top-k 与置信度；可选导出 ONNX/TorchScript。
9. 文档与示例：`configs/example.yaml`、数据集说明文档、`requirements.txt` 补充依赖。

## 位置与运动特征计算
### 1 每帧的几何特征
先归一化到 [0,1]，做成无关分辨率的量：
- cx_t = (x_t + 0.5 * w_t) / W （中心点 x）
- cy_t = (y_t + 0.5 * h_t) / H （中心点 y）
- sw_t = w_t / W （相对宽度）
- sh_t = h_t / H （相对高度）
- ar_t = w_t / (h_t + ε) （长宽比，最好 log 一下）
- area_t = (w_t * h_t) / (W * H) （相对面积）
这 6 维就是每帧的「静态几何特征」。

### 2 运动特征（几何的时间差）
让 TCN 能看到「变化」而不是绝对值，加一阶、二阶差分：
- 速度：
  - vx_t = cx_t - cx_{t-1}
  - vy_t = cy_t - cy_{t-1}
  - vs_t = sw_t - sw_{t-1}（尺度变化）
  - vh_t = sh_t - sh_{t-1}
- 加速度：
  - ax_t = vx_t - vx_{t-1}
  - ay_t = vy_t - vy_{t-1}
- 再衍生几个标量，帮助区分“抖、晃、飘、稳”：
  - 速度大小：speed_t = sqrt(vx_t^2 + vy_t^2)
  - 尺度变化率：scale_rate_t = (sw_t - sw_{t-1}) / (sw_{t-1} + ε)
到这里每帧一个几何+运动特征向量，
```python
g_t = [cx_t, cy_t, sw_t, sh_t, ar_t, area_t,
       vx_t, vy_t, vs_t, vh_t,
       ax_t, ay_t,
       speed_t, scale_rate_t]   # 14 维左右
```
注意几点：
- 第一帧没有前一帧，可以直接置 0。
- 所有特征最好做一下标准化（减均值 / 除方差），用训练集统计。

## 架构设计细节

### 数据流程
- 输入：序列文件夹（最多200帧图片 + JSON元数据）
- 帧采样：均匀采样固定数量帧（如64帧）
- 特征提取：
  - **视觉特征**：2D CNN 对每帧提取空间特征
  - **位置特征**：从 JSON 读取 (x,y,w,h)，归一化并编码
  - **运动特征**：计算相邻帧间的位置变化（速度、加速度）
- 时序建模：TCN 处理特征序列
- 分类输出：全连接层输出类别概率

### 模型组件
1. **FrameEncoder (2D CNN)**
   - 输入：单帧图像 (C, H, W)，H,W ≤ 64
   - 输出：特征向量 (D,)
   - 备选：自定义小型 CNN / MobileNetV2 / RESNET-18 轻量变体

2. **PositionEncoder (MLP)**
   - 输入：位置与运动特征向量 (T, 14)
   - 输出：位置与运动特征嵌入 (T, D_pos)

3. **TemporalConvNet (TCN)**
   - 输入：融合特征序列 (T, D+D_pos)
   - 输出：时序特征 (D_tcn,)

4. **Classifier (FC)**
   - 输入：TCN 输出或全局池化后的特征
   - 输出：类别概率 (num_classes,)

### JSON 元数据格式（假设）
```json
{
  "resolution": {
    "width": 960,
    "height": 544
  },
  "tracks": [
    {
      "image": "00000000.jpg",
      "bbox": {
        "x": 463,
        "y": 264,
        "w": 22,
        "h": 18
      }
    },
    {
      "image": "00000001.jpg",
      "bbox": {
        "x": 464,
        "y": 264,
        "w": 23,
        "h": 19
      }
    },
    ...
  ],
}
```

### 文件结构规划
```
dataset/110_video_frames_2025_11_all_200_have_json/
├── bird/
│   ├── seq_0001/
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ...
│   ├── seq_0001.json
│   └── ...
├── drone/
│   └── ...
├── night_drone/
│   └── ...
├── plane/
│    └── ...
├── train.csv                # 训练集列表
├── val.csv                  # 验证集列表
└── test.csv                 # 测试集列表

src/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Dataset类，加载序列
│   └── transforms.py       # 数据增强
├── models/
│   ├── __init__.py
│   ├── feature_extractor.py  # CNN特征提取器
│   ├── classifier.py        # 完整分类模型
│   └── tcn.py              # 将model_cnn_tcn.py移到这里并重命名
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # 评估指标
│   └── visualization.py    # 可视化工具
└── main.py                 # 训练/推理入口

scripts/
├── train.py                # 训练脚本
├── eval.py                 # 评估脚本
└── infer.py                # 推理脚本

configs/
├── default.yaml            # 默认配置
└── example.yaml            # 示例配置

tests/
├── conftest.py
├── data/
│   └── test_dataset.py
└── models/
    └── test_classifier.py
```