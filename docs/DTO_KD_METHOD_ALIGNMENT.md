# DTO-KD 方法对齐与版本界限（2026-11-07 更新）

论文：Hayder 等，DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation，ICLR 2026。
正式 PDF：https://proceedings.iclr.cc/paper_files/paper/2026/file/e546ce4230c15bc422a500bb47b341ea-Paper-Conference.pdf
来源哈希、公式位置和作者代码检索范围见 `assets/DTO_KD_REFERENCE_SOURCE.json`。
目前未定位到可核验的作者官方实现，因此这里是**论文公式级实现及 TS-FSAR 适配**，不能声称作者代码或完整系统复现。

## 当前可区分的版本

| 版本 | 两个目标 | 权重求解 | 辅助 CE | KL 教师目标 |
|---|---|---|---|---|
| E0 | 原 DCM、原 KL | 官方固定系数 | 原系数 | 原双向梯度 |
| E0_detach | 同 E0 | 同 E0 | 原系数 | detach，教师 CE 仍训练 |
| 历史 E1 | log(KL)、log(DCM) | legacy_logits 或 v2 Eq.18，必须读取运行配置区分 | 原系数 | 原双向梯度 |
| E1_safe | log1p(基础加权 KL)、log1p(全部监督 CE) | Eq.18 更新形式 | 合并后共同缩放 | detach，教师 CE 仍训练 |
| E1_amortized | log(KL)、log(DCM) | Eq.18 直接 pi 更新再 softmax | 原系数，独立相加 | detach，教师 CE 仍训练 |
| E1_exact | log(KL)、log(DCM) | Eq.8--15，逐 episode 真实共享 LSN 梯度的 simplex 闭式解 | 原系数，独立相加 | detach，教师 CE 仍训练 |

E1_safe 的 log1p 和监督分组是额外设计，**不是论文的原始 log-loss**。
E1_amortized 与 E1_exact 两个新配置不使用这些改动，不把旧 E1_safe 的证书、训练状态或结论移植过来。
原有运行配置和原有默认队列不被静默替换；新版本只能通过新实验名显式启动。

## 论文公式与实现

记 d=KL_GLAC，t=CE_DCM。数值保护使用 log(max(loss,1e-8))；它不是 log1p。
低于 floor 时导数为零，此时闭式解可能落在边界并使两目标组合驻留；如实记录，不通过隐含回退系数或梯度裁剪掩盖。
对显著为负或非有限损失立即报错，不把它们当作正常 KL。

论文 Eq.8、9：

    g_d = grad_shared_LSN log(d)
    g_t = grad_shared_LSN log(t)
    pi* = argmin_{pi>=0, sum(pi)=1} ||pi_d*g_d + pi_t*g_t||²

论文 Eq.11--15 的两目标解（受 simplex 约束）：

    pi_d = clip((g22-g12)/||g_d-g_t||², 0, 1)
    pi_t = 1-pi_d

`experiments/dto_kd/exact.py` 每步用两次 autograd.grad 取得真实梯度，CPU FP64 计算 Gram 与梯度差范数；
仅纳入两目标都在计算图上连接的可训练 `model.side.*` 参数，逐项记录名称和数量。
等梯度时选择 .5/.5；边界解保留，不以 softmax 替代闭式 simplex 解。
通过 KKT 条件检查 `<g_i,g*> >= ||g*||²`；内部解检查自然包含两目标等贡献性质。
论文中的闭式理论和 Eq.18 近似必须区分，后者没有逐步读取真实梯度夹角。
不能将论文的所有范数界或严格下降结论无条件套用到反向共线、边界、零梯度等情形。

论文 Eq.18：

    pi_next = softmax(pi - eta * (pi dot log_losses) * log_losses)

`controller.py::update_from_logs` 的 paper_pi_softmax 分支直接实现上述形式，没有 logit Jacobian 或 logit clipping。
E1_amortized 当前 episode 使用旧 pi，下一 episode 使用更新后的 pi；AMP 跳步回滚控制器。
E1_exact 每步重算 pi，没有持久的控制器状态。

二者的实际模型目标都是：

    auxiliary = CLS_ALPHA*student_CE + KD_ALPHA*teacher_CE + SIDE_CLS_ALPHA*side_CE
    L = auxiliary + pi_d*log(d) + pi_t*log(t)

KL 的基础系数不再作为固定乘子嵌入两目标组合；有效系数为 pi_d/d，DCM 为 pi_t/t。
不能声称新参考版继承了 E1_safe 的固定 KL 系数上界。
闭式版直接用实际 log 梯度几何求权重以处理该尺度，近似版保留原论文近似的局限和小 KL 风险作为可检验问题。

## 对齐范围与不能省略的限制

- **共享空间**：只在共享 LSN 上求两目标权重；TSM、私有分类头、教师 Adapter 不进入该 Gram。权重随后作用于各目标的完整计算图。
- **教师训练**：KL target detach，教师 Adapter 仍按原 KD_ALPHA 的教师 CE 更新；骨干冻结。不同于论文的完整固定教师。
- **辅助项**：原 CE 的绝对系数保持，但它们不属于该两目标最小范数问题。加上辅助项后可能再次出现负投影。
- **优化器**：保留官方 AdamW/AMP/学习率。原梯度的双目标几何性质不自动保证经过 AdamW 动量、预处理及 weight decay 后的参数更新也同时下降。
- **数值**：Gram 在 CPU FP64 累积，原模型仍采用原 AMP；不等同于全模型 FP64 求导。没有添加任意 KL 上限、额外梯度裁剪或新的网络前向。
- **资源**：闭式版每步增加两次梯度遍历，CPU saved-tensor offload 在该模式每步启用，速度与 CPU 内存/PCIe 开销必须报告。
- **证据**：局部负余弦与尺度差异是观察，尚未证明它们导致最终泛化损害。精确求解也不保证性能提升。

`gradient_groups.jsonl` 分别记录原始/实际加权范数与余弦、KL/DCM 两目标组合的投影，以及加入辅助 CE 后总梯度的投影。
负点积表示沿负梯度走对该目标是一阶上升方向；这些是 AdamW 前的诊断，不伪装成实际优化器更新。

## 验证

`tests/test_dto_kd_reference_cpu.py` 覆盖：闭式解与独立密集网格最小值及 KKT 对照、极端公共尺度、边界/反向/等梯度、
真实共享梯度构造、不污染 .grad/RNG、两种 shot 的原始 TS-FSAR 损失路径、教师 KL 零梯度与原教师 CE 强度、
有效系数重建、Eq.18 独立手算、公共训练循环及无控制器 checkpoint、证书与独立配置。
CPU 使用合成视频/小 LSN；不运行原服务器 GPU、真实 CLIP 推理或训练。
校园显式 `--dto-reference-checks` 重跑 E0 一步等价及两个新模式的 32-episode CUDA 短验收；没有校园通过记录前不得声称 A10 已验证。
