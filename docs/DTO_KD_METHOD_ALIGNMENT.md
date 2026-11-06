# DTO-KD 方法对齐说明

论文：Zeeshan Hayder 等，**DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation**, ICLR 2026。正式论文：[proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2026/file/e546ce4230c15bc422a500bb47b341ea-Paper-Conference.pdf)。

## 公式与代码

论文把蒸馏损失和任务损失写成 `L_distill`、`L_task`，并在 `J=[∇log L_distill, ∇log L_task]` 上求解
`argmin_{π∈Δ} ||Jπ||²`。两目标闭式解为：

```
π_distill = (g22 - g12) / (g11 + g22 - 2 g12)
π_task    = (g11 - g12) / (g11 + g22 - 2 g12)
```

实现位置：`experiments/dto_kd/controller.py::closed_form_weights`。它对退化分母做 0.5/0.5 回退，并将结果裁剪到 simplex；CPU 单元测试覆盖共线和冲突梯度。

论文的实际训练建议是 amortized update（Eq. 18）：以当前 `π` 计算
`0.5 (π_distill log L_distill + π_task log L_task)^2` 的权重梯度，再经 softmax 保持 simplex。实现位置：`DTOController.update`。TS-FSAR 的 E1 使用一次反向传播的近似：模型反向使用 `π_distill log KL + π_task log CE_DCM`，控制器用 detached loss 更新下一 episode 的权重。这样没有把普通 KL 调系数、PCGrad 或 GradNorm 冒充 DTO-KD。

论文中的 Gradient Conflict 是 `<g_distill,g_task><0`；Gradient Dominance 由 `||g_distill||/||g_task||` 描述。`experiments/dto_kd/controller.py::gradient_metrics` 和 `gradient_utils.py` 记录余弦、负冲突比例、投影和范数比。训练每 `DTO_KD.GRADIENT_SAMPLE_INTERVAL` 个 episode 才额外调用 `autograd.grad`，不会替代正常 AMP backward。

## TS-FSAR 适配

- `L_task = CE_DCM`，由既有 `TS_DCM` score 计算。
- `L_distill = KL_GLAC`，由既有 `self.kl` 计算。
- `CE_global_ADC`、`CE_LSN_text`、`CE_teacher_CLIP_text` 保留在 `auxiliary` 中。
- 教师 Adapter 仍由原图反向传播；没有在适配器中隐式 detach。
- CLIP 主干仍按原 `freeze_backbone` 冻结；TSM、LSN、Adapter 的参数分组仍由原优化器决定。
- E2 只替换 KL 的固定系数，教师文本 CE 仍使用原 `KD_ALPHA`；候选 0.05/0.1/0.2 只能由 Meta-Val 选择。
- E3 使用 Meta-Train 初始梯度范数比的中位数产生一个固定 KL 系数；`scripts/calibrate_dto_scale.py` 不读取 Meta-Test。
- E4 只保留 `episode_aware` 接口并明确抛出 `NotImplementedError`。

## 与论文的差异和边界

论文的示例模型有单独的任务/蒸馏 head、冻结教师和完整双梯度闭式求解。TS-FSAR 的 GLAC 教师 Adapter 默认可训练，且还有三个辅助 CE；因此本实现是**忠实的 amortized DTO-KD 适配版**，不是论文图像分类/检测代码的逐行复现。它没有改变 TS-FSAR 的蒸馏对象、网络结构、episode 采样和推理融合。闭式解仅作为独立诊断工具，E1 训练使用论文 Eq. 18 的单反向近似。

尚未在本机 GPU 或校园 GPU 上验证 CUDA 训练、显存开销和最终准确率；CPU 测试只覆盖公式、数值稳定性、状态恢复和梯度传播。
