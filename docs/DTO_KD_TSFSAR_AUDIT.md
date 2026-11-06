# TS-FSAR 损失与参数共享审计

当前 `trainers/ts_fsar.py` 的训练损失为：

```
VIS_ALPHA * CE_DCM
+ CLS_ALPHA * CE_global_ADC
+ SIDE_CLS_ALPHA * CE_LSN_text
+ KD_ALPHA * (KL_GLAC + CE_teacher_CLIP_text)
```

`CE_DCM` 来自 `TS_DCM` 的 episode query-support score；`CE_global_ADC` 使用 α-D 表示和固定 64 个 base class；`CE_LSN_text` 使用 LSN CLS 与文本 embedding；`CE_teacher_CLIP_text` 使用 Adapter 后 CLIP CLS 与文本 embedding；`KL_GLAC` 是 ADC logits 与 Adapter-CLIP logits 的 `DistillKL`。

`utils/optimizer.py` 保持原参数分组：`side*`、`Linear/up_linear/up_ln`、`generator*` 和 `Adapter*` 可训练，其余（包括 CLIP 主干）冻结。因而：

- LSN side/up 和 TS-DCM generator 同时收到 DCM、global CE、LSN text CE 及其通过 α-D 路径的蒸馏影响；
- Adapter 收到教师文本 CE 和 KL 的梯度，未被 DTO 适配器隐式 detach；
- global `Linear` 主要收到 ADC CE 及 KL 学生端梯度；
- 冻结 CLIP 主干不接收更新；
- DTO 只把 `CE_DCM` 与 `KL_GLAC` 作为两目标，其他三个监督项保留。

`experiments/dto_kd/model.py` 用 forward hooks 读取已经计算的 score/CE/KL，避免复制或改写 TS-FSAR 的前向结构。E0 不导入该 wrapper；`DTO_KD.MODE=off` 也直接返回原始 loss。正式 CUDA 训练前应在校园平台用同一初始化和合成 batch 做一步梯度/参数差异检查。
