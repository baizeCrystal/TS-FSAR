# 校园 DTO 实验配置与研究有效性审计（2026-11-07）

本次审计由用户询问“是否符合官方配置、会不会得到无效数据”触发。只做 CPU、文件及论文检查，不启动原服务器 GPU，不操作校园进程和既有输出。

## 结论

不能将截至 `62d6f01` 的全部校园实现称为“官方配置的严格复现”或“已完成研究验收”。存在已确认的 5-shot 学习率偏差、DTO 权重更新参数化差异，以及未完成的记录/验证要求。Smoke 成功只证明所覆盖步骤能运行。现有运行应按实际配置保存，不能将其性能直接与历史 60.774%/70.696% 相减来归因 DTO。

## 原始配方与当前配置

核对基准是本仓库保留的发布文件 `configs/ssv2_small/Metatrain_{1,5}shot.yaml` 与 `configs/base.yaml` 的合并结果。5-shot 发布配置最初引入提交为 `e9aa1fe`，其中三组基础 LR 均为 `0.0003`。额外读取原服务器已完成历史复现的 `output/table_reproduction/ssv2small_5shot_seed0_gpu3_restart_20261105_180156/ssv2_small_5shot/train.yaml`，三组 LR 同为 `0.0003`；其 best_result.json 为 70.69600124672056%，episode 2500。

| 设置 | 1-shot 发布配方 | 5-shot 发布配方 | 校园 `62d6f01` |
|---|---|---|---|
| WAY、帧数、训练 query/类 | 5、8、5 | 5、8、5 | 相同 |
| SHOT | 1 | 5 | 各自正确 |
| 训练 episode/评估 episode | 10000/10000 | 10000/10000 | 相同；smoke 是独立短配置 |
| episode batch / 梯度累计 | 1/1 | 1/1 | 相同 |
| 验证间隔 | 500 | 500 | 相同 |
| AdamW、eps、weight decay | AdamW、1e-6、0.1 | 相同 | 相同优化器实现 |
| GENERATOR/LSN/LSNHEAD LR | 0.00025 | 0.0003 | 1-shot 正确；5-shot 错为 0.00025 |
| 原 KL / 教师 CE 系数 | 0.1/0.1 | 0.01/0.01 | E0 相同；E2 仅单独改 KL |
| 学生全局 CE / LSN CE 系数 | 2/0.1 | 2/0.01 | 相同 |
| 推理融合 Vision/CLIP/Side | 0.4/0.2/0.4 | 0.8/0.05/0.15 | 相同 |

本次将所有六份 5-shot DTO 配置三组基础 LR 修正为 `0.0003`，不改已生成的校园 `config.yaml`，不改运行中学习率。正式 5-shot 若要求与发布配方对齐，必须在新目录从相同初始化重新完整训练；旧 `0.00025` 结果只能单列探索性对照。此次未修改训练算法、损失或模型代码。

`campus_protocol_62d6f01.json` 保存修正前配置哈希及差异；`campus_protocol_corrected_configs.json` 保存修正后检查（其 code_commit 是生成报告时的基底提交，配置本身以 SHA256 标识）。两份报告均只认证配置层面的项目。

## 选模协议的有意差异

历史结果使用 24 类 Novel Test 周期评估并选 best；当前根据用户要求用独立 12 类 Meta-Val 选 best，最终再用 24 类 Test。Train/Val/Test 源划分保持 64/12/24 类；Val 的 `valN` 仅改前缀为旧 loader 需要的 `testN`，不改视频或类别编号。Val 文本为固定类别名，最终 Test 仍是原 LLM 描述。这是新的选模协议，不是复现历史 Test-best 的同一个统计量。需要以当前协议重新训练的 E0 作为 E1/E2 对照。

## DTO 原论文核对

重新下载并读取正式论文第 7 页、Algorithm 1 和 Eq.18：
https://proceedings.iclr.cc/paper_files/paper/2026/file/e546ce4230c15bc422a500bb47b341ea-Paper-Conference.pdf

论文写的是对权重 π 更新，再经 softmax 归一化。当前代码对 softmax 的 logits 更新，包含 Jacobian，且有 logit clipping。这是参数化适配，不等价于论文文字的逐步更新。闭式双梯度权重函数并没有用于当前 E1 训练。TS-FSAR 另保留三个辅助 CE 以及可训练教师 Adapter；因此不能直接宣称论文的方向保证在本实验成立。完整公式差异见 `../DTO_KD_METHOD_ALIGNMENT.md`。

为了不改变已经运行的实验定义，本次没有静默切换控制器公式。现有 E1 应明确标注为 DTO-KD-inspired softmax-logit 适配版；若要报告严格公式版，需要独立版本和从头训练。

## 已验证及仍未完成

新增 CPU 测试使用真实 `TS_FSAR`、`DTOTSFSAR`、TS-DCM、原始 KL 和辅助 CE，但用小型合成 LSN 替换 CLIP/PCA 特征。两种 shot、关闭适配与静态 KL 等于原系数的四种组合，核对总损失、所有参数梯度、教师 Adapter 梯度及 AdamW 一步参数。通过该测试只覆盖这些真实损失路径，不是完整视觉模型 CUDA 逐步一致性证明。

已有 CPU 测试还验证控制器的实际 logits 更新、诊断不写入 `.grad`、CPU activation 保存的合成多步等价。用户已确认校园 E1 两个 shot 的三步 smoke（包括诊断和短验证）通过。GPU 7 后续出现 uncorrectable ECC，属于另一项硬件失败，不能记作完成的训练。

本次验证命令为 `CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_dto_kd_cpu tests.test_campus_execution_cpu tests.test_dto_kd_memory_cpu tests.test_campus_protocol_cpu tests.test_dto_kd_cuda -v`：共 28 项，24 项 CPU 通过，4 项需要显式启用的 CUDA 测试跳过。配置审计修正前因六份 5-shot 配置各三项 LR 差异返回 1，修正后全部十二份配置返回 0。语法检查与 `git diff --check` 通过。

以下用户原始要求尚未完整实现/验证，必须在研究报告中披露：

- 全模型 CUDA baseline/关闭适配一步损失、梯度和参数更新等价认证。
- 实际 Train/Val/Test episode 序列、视频/帧索引及序列哈希。相同 seed 不等于已经证明逐 episode 配对。
- 共享 LSN 与各参数组、实际加权、辅助总梯度、有效更新/裁剪等诊断。当前 CSV 仅含全体可训练参数的原始 KL/DCM 向量指标。
- 验证/测试各分支（matching、LSN、Adapter、融合）的逐 episode 预测与 Bootstrap 输入。当前普通 test() 只输出完整融合的汇总值。
- 完整 optimizer/scaler/controller/RNG/episode 恢复。当前 best 只有模型权重，不能断点续训。
- E3 当前早期原始全参数范数校准不等于已经匹配 E1 的实际加权共享 LSN 梯度尺度，默认采样间隔 500 与 max_step 500 还只会使用首个采样点。

这些缺失不等于所有预测都错误，但限制了可提出的结论。当前不能依据这批数据宣称严格复现 DTO-KD、证明梯度冲突被解决、或排除梯度尺度与其他因素的因果作用。

## 校园端只读核对

```bash
python scripts/audit_campus_protocol.py
python scripts/audit_campus_protocol.py \
  --run-root runs/campus/a10_seed0_20261107_103208 \
  --output /data/fsar_workspace/campus_protocol_original_runs.json
```

脚本不导入 torch，不执行模型，不更改原运行。输出文件必须不存在。原队列 5-shot 配置预期因旧 LR 返回非零；报告指出具体字段。它不检查视频解码、全模型数值、实际 episode 配对或最终精度。
