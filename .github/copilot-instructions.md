# Scene Graph Benchmark - Copilot 指南

## 项目概述

基于 maskrcnn-benchmark 的场景图生成 (SGG) 框架，核心论文 "Unbiased Scene Graph Generation from Biased Training" (CVPR 2020 Oral)。本项目扩展了几何校准和噪声建模方法用于改进长尾关系分类。

## 架构要点

### 核心模块路径
- **关系预测头**: `maskrcnn_benchmark/modeling/roi_heads/relation_head/`
- **预测器实现**: `roi_relation_predictors.py` (MotifPredictor, VCTreePredictor, IMPPredictor, TransformerPredictor)
- **评估逻辑**: `maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py`
- **配置系统**: `maskrcnn_benchmark/config/` + `configs/*.yaml`
- **校准方法**: `test_mixture_vs_single.py`, `noise_transition_model.py`

### 三种评估协议
```python
# PredCls: GT boxes + GT labels (最常用于校准研究)
cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = True
cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = True

# SGCls: GT boxes only
cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = True
cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False

# SGDet: 完全检测
cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
```

## 数据流

1. **数据加载**: `maskrcnn_benchmark/data/` → VG HDF5 文件 (`datasets/vg/VG-SGG-with-attri.h5`)
2. **特征提取**: Faster R-CNN backbone → RoI features
3. **关系预测**: `relation_head.py` → predictor → `pred_rel_scores` [N_pairs, C]
4. **后处理校准**: Calibrator.calibrate() → 调整概率分布
5. **评估指标**: R@K, mR@K, zR@K 在 `sgg_eval.py` 计算

## 常用命令

### 训练关系模型
```bash
python tools/relation_train_net.py --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
  MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```

### 评估
```bash
python tools/relation_test_net.py --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
  MODEL.WEIGHT path/to/model.pth TEST.IMS_PER_BATCH 1
```

### 校准方法对比测试
```bash
# 完整 battle (Baseline vs Single vs Mixture vs NTM)
python test_battle.py --num-test 100 --train-ntm --ntm-epochs 10

# 单独测试 Mixture vs Single
python test_mixture_vs_single.py --num-test 100 --gamma 0.5 --eta 0.3
```

## 项目特有模式

### 添加新预测器
1. 在 `maskrcnn_benchmark/modeling/roi_heads/relation_head/` 创建 `model_xxx.py`
2. 在 `roi_relation_predictors.py` 注册:
```python
@RELATION_PREDICTOR_REGISTRY.register("XXXPredictor")
class XXXPredictor(nn.Module):
    def __init__(self, config, in_channels):
        ...
    def forward(self, proposals, rel_pair_idxs, ...):
        return obj_dists, rel_dists, ...
```

### 几何特征标准 (9维)
```python
# compute_geometric_features(box_s, box_o, img_w, img_h)
[Δx/w_s, Δy/h_s, log(w_s/w_o), log(h_s/h_o), log(area_s/area_o),
 IoU, cos(θ), sin(θ), normalized_dist]
```

### 校准器接口
```python
class Calibrator:
    def calibrate(self, rel_probs, boxes, rel_pairs, obj_labels, img_size):
        """
        Args:
            rel_probs: [N, C] 原始关系概率
            boxes: [M, 4] 边界框 (xyxy)
            rel_pairs: [N, 2] 关系对索引
            obj_labels: [M] 物体类别
            img_size: (width, height)
        Returns:
            calibrated_probs: [N, C] 校准后概率
        """
```

### 噪声转移模型 (Forward Correction)
```python
# noise_transition_model.py
# 训练时: p_obs = p_clean @ T, loss = -log p_obs[c̃]
# 推断时: pred = argmax p_clean (使用干净预测)

from noise_transition_model import NoiseCorrectedModel, Config
cfg = Config()
cfg.K = 8  # 混合成分数
model = NoiseCorrectedModel(cfg)
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `noise_transition_model.py` | 标签噪声转移矩阵模型 (Forward Correction) |
| `test_mixture_vs_single.py` | Single/Mixture Prototype 校准器对比 |
| `test_battle.py` | 多方法 Battle 测试脚本 |
| `test_geo_performance_v3.py` | Mixture Prototypes 详细实现 |
| `configs/e2e_relation_X_101_32_8_FPN_1x.yaml` | 主配置文件 |

## 注意事项

- **Conda 环境**: `torch50` (PyTorch 1.4, CUDA 10.1)
- **GloVe 词向量**: 项目根目录需要 `glove.6B.200d.pt`
- **预训练模型**: `VCTreePredictor/VCTreePredictor/PredCls/model_best.pth`
- **数值稳定**: 所有 log/softmax 操作使用 `eps=1e-12` 防止 NaN
- **长尾问题**: 使用 mR@K 而非 R@K 评估尾部类别性能

## 开发规范

### 张量形状注释
```python
# 始终标注关键张量形状
p_clean = F.softmax(logits, dim=1)  # [B, C]
T = transition_module(e_pair)        # [B, C, C]
p_obs = torch.bmm(p_clean.unsqueeze(1), T).squeeze(1)  # [B, C]
```

### 配置优先级
```
命令行参数 > YAML 配置 > defaults.py
```

### 测试新方法
1. 先在 `test_mixture_vs_single.py` 中添加
2. 集成到 `test_battle.py` 进行对比
3. 确保实现 `Calibrator.calibrate()` 接口
