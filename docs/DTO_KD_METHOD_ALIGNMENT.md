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
`0.5 (π_distill log L_distill + π_task log L_task)^2` 对权重的梯度，更新 `π`，再经 softmax 保持 simplex。当前 E1 在 `DTOController.update` 中采用 **softmax-logit 参数化的适配**：令 `π=softmax(u)`，对 `u` 求导并更新 `u`，包含 softmax Jacobian。因此它与论文文字所述的直接 `π` 更新再 softmax **不完全相同**，不能声称 Eq.18 逐步严格复现，也没有作者代码逐行一致性验证。模型反向使用 `π_distill log KL + π_task log CE_DCM`，控制器用 detached loss 更新下一 episode 的权重。

具体而言，令 `s=π·log L`，论文文字更新为 `π_next=softmax(π-η*s*log L)`；当前实现为 `u_next=clip(u-η*(diag(π)-ππᵀ)*(s*log L))`，随后 `π_next=softmax(u_next)`。两者的数值路径不同。既有 CPU 控制器测试验证的是后者的实现自洽性，不是前者的一致性。2026-11-07 已重新读取正式 PDF 第 7 页的算法与 Eq.18 确认这一差异；未在运行中的实验里切换控制器。

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

论文的示例模型有单独的任务/蒸馏 head、冻结教师，并讨论双梯度闭式求解及单反向近似。TS-FSAR 的 GLAC 教师 Adapter 默认可训练，且还有三个辅助 CE；当前实现应标为 **DTO-KD-inspired softmax-logit 适配版**，不是论文算法的严格复现。它没有改变 TS-FSAR 的蒸馏对象和网络结构。闭式解仅作为独立工具，E1 训练使用上述单反向适配。额外辅助损失、可训练教师和近似求解也意味着不能直接套用论文关于最优方向的理论保证。

校园平台已由用户确认 E1 两种 shot 的三步 smoke 通过，不能等同完整训练或全模型 CUDA 一步一致性认证。本机只运行 CPU 合成测试。当前梯度 CSV 仅含全体可训练参数的原始梯度范数、余弦和投影；尚未记录共享 LSN 分组、实际加权梯度、完整辅助梯度或裁剪前后指标。完整优化器/AMP/RNG/控制器断点恢复、真实 episode 序列哈希、各推理分支逐 episode 预测记录也尚未实现。详见 `audits/CAMPUS_PROTOCOL_AUDIT_20261107.md`。
