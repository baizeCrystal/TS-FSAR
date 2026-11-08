# SAM＋QAM：冻结 E0 的类别条件通道匹配

代码分支：<https://github.com/baizeCrystal/TS-FSAR/tree/research/sam-qam-channel>。

实现 TDM（CVPR 2022）的 Support Attention Module 和 Query Attention Module，迁移到 TS-FSAR 的 256 维 LSN CLS/Patch → α-D → TSM 匹配路径。本轮只比较 E0 与这一个 SAM＋QAM 方案。没有接入 DTO-KD，没有 PS/DS、SVD 或测试时优化。

## 实验定义

- 分别加载历史正式 1-shot / 5-shot E0 checkpoint。冻结其全部参数和 buffer 更新，始终 `eval()`；不做 PCA 重初始化。
- 只训练新 SAM/QAM：在官方 64 类 Base Train 上采样 5-way、对应 shot、每类 5 Query、每视频 8 帧；沿用原训练视频预处理。
- 新模块训练 10,000 Episodes，matching CE，AdamW、初始 LR 0.00025、weight decay 0.1、无 warmup 的 cosine。它是冻结 E0 的适配实验，不是从头复现 TS-FSAR 或 TDM 的完整训练。
- 按固定训练预算使用最终模块，不读 Novel-Test 来选 SAM/QAM checkpoint。原 E0 checkpoint 本身具有历史选模来源，不能把本次结果解释为独立确认集验证。
- 正式 Meta-Test 完全复用已有固定 Episode Manifest：官方 24 个 Novel Test 类别，5-way、1/5-shot、每类 1 Query、8 帧、10,000 Episodes。
- E0/E1 使用同一像素输入、同一次冻结编码器前向、同一 E0 checkpoint。只替换 matching score；CLIP、side 分支和原始三分支乘积融合保留。
- 首轮不堆叠旧 CTM 模块；SAM/QAM 本身承担通道选择。

## 方法与原论文的一致性

原论文：[Task Discrepancy Maximization for Fine-Grained Few-Shot Classification, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Task_Discrepancy_Maximization_for_Fine-Grained_Few-Shot_Classification_CVPR_2022_paper.html)。

核对作者代码的固定版本：[models/TDM.py @ 24be5f3](https://github.com/leesb7426/TDM/blob/24be5f3ee75e10f74e07bedfce28fb51a001f7ab/models/TDM.py)，SHA256 `1b0f6d5e0b7233e439db6c2fe6b12637a2dc87f505aaf15f58ff34d357bbf1f7`。本目录独立实现公式；没有复制作者源码。`tests/author_oracle.json` 是运行该固定作者版本得到的小型数值对照，包含输入/参数生成说明。

保留 Eq.2–12：

1. Support 按标签构造 shot 均值原型。
2. `intra` 是各通道响应到本类通道均值响应图的平均平方距离，**不是样本类内方差**。
3. `inter` 是该通道响应到其他类别均值响应图距离的最小值。
4. 独立 `FC(C,2C,bias=False) → BN → ReLU → FC(2C,C,bias=False) → 1+tanh` 生成 intra、inter、Query 权重；Linear 用 Xavier uniform。
5. SAM 内混合系数 α=0.5，SAM/QAM 混合系数 β=0.5，采用论文固定值。
6. 按原文 Sec.4.2，训练时给最终权重加 uniform[-0.2,0.2] 噪声并截断到 [0,2]；测试关闭噪声，BN 使用训练所得 running statistics。

视频迁移设计明确为：输入 Patch `[N,8,256,196]`，保持 8×196 坐标，按 shot 求原型，代表性距离对 time×patch 求均值。每个 `(Query,候选类)` 得到一个 `[256]` gate，全部八帧共用；没有帧平均后再匹配，也没有绝对值、ReLU 或新增特征标准化改变原始有符号 LSN 表示。

候选类 c 的 gate 同时缩放 Query 和本类 Support 的 CLS/Patch，然后重新执行原始 signed CLS–Patch 权重、α-D、每类 shot-mean 描述符、原 `cos_sim`（norm product + 0.01）和 signed TSM 聚合。TSM 生成器始终读取**未缩放**的原 CLS 条件。返回单个 `[num_query,5]` 分数。

`SAMQAM.forward` 和 `matching` 都没有 Query 标签参数；Query 标签仅进入 Base Train 的 matching CE，或 Test 预测后的准确率统计。测试逐 Query 的门控不依赖其他 Query：没有 Query 集合伪标签或 transductive 更新。

## 校园端一键启动

复用已有环境 `/data/fsar_workspace/envs/tsfsar-dto-kd`、SSv2-Small 视频和 `/data/fsar_workspace/psds_assets/` 的资源。不需要新建 GitHub 仓库、不需要重新上传数据或权重、不需要安装新依赖。完整运行环境清单见项目根目录 `requirements-campus.txt`。

```bash
(
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd
export PATH="$HOME/.local/gh-cli/bin:$HOME/.local/bin:$PATH"

TREE=/data/fsar_workspace/TS-FSAR-SAMQAM
if [[ ! -e "$TREE" ]]; then
  git clone --single-branch --branch research/sam-qam-channel \
    https://github.com/baizeCrystal/TS-FSAR.git "$TREE"
fi
cd "$TREE"
test "$(git branch --show-current)" = research/sam-qam-channel

bash research/sam_qam_experiment/launch_campus.sh --gpus 4,5
)
```

准备脚本先运行 CPU 数值测试，再通过已有私有 Release 的 SHA256 校验并复用资源，只在新 clone 创建 CLIP 链接和解压固定 Manifest。需要此前已登录的 `gh`；若提示未登录，先用 `gh auth login --hostname github.com --git-protocol https --web` 完成授权。不会将凭据写进脚本或项目。

后台启动前检查两张卡的现有进程和 ECC；GPU 忙时拒绝启动，不停止任何已有进程。GPU4 跑 1-shot，GPU5 跑 5-shot；可用 `--gpus 4,6` 等显式指定。GPU0–3 不受影响，GPU6 默认不用。

每组依次执行：

1. 在两个 Base Train Episodes 上做可丢弃的新模块训练检查。
2. 用这个临时模块做三个固定 Test Episodes 的工程检查：关闭模块位级 E0 一致、全 1 gate 数值还原、完整融合一致、无 NaN/Inf、模型状态未修改。
3. **重新初始化**正式 SAM/QAM，在 10,000 个 Base Train Episodes 上训练，不继承临时训练或测试状态。
4. 正式模块依次评测固定 3 / 200 / 10,000 Episodes。200 仅作探索记录，不自动改配置。
5. 两种 shot 成功后自动执行 10,000 次 Episode 配对 Bootstrap，生成表格、纠正/新增错误、Hard Negative 和耗时报告。

临时模块严格禁止用于 explore/formal。任何阶段失败会停止该组后续阶段，另一组可继续。小检查不保证实际 A10 全程成功；本服务器只完成 CPU 检查，首次真实 GPU 验证由上述校园阶段执行。

## A10 显存与耗时

固定 episode batch=1，训练仍为每类 5 Query；不为省显存修改 way、shot、Query 数或输入帧数。

冻结编码器每次处理 4 个完整视频（32 帧）。候选类匹配按 1 个 Query 分块，使用 PyTorch non-reentrant activation checkpointing 重计算 α-D；SAM/QAM 的 BN 与噪声在分块外仅执行一次。匹配和新模块训练均为 FP32，未启用 AMP。没有完整编码器反向和大模型 AdamW 状态；额外计算仍需校园实测，日志会给出实际峰值显存和耗时。

训练日志每 100 Episodes 输出一次该区间平均 CE/accuracy，另存每 Episode 记录，避免把单 Episode 难度波动误认为训练趋势。检查/测试日志每 10 Episodes 输出。

## 日志与结果

```bash
RUN="$(cat /data/fsar_workspace/latest_tsfsar_samqam_root.txt)"
tail -n 40 -F "$RUN/shot1.console.log" "$RUN/shot5.console.log"
```

只看 1-shot：

```bash
RUN="$(cat /data/fsar_workspace/latest_tsfsar_samqam_root.txt)"
tail -n 40 -F "$RUN/shot1.console.log"
```

只看 1-shot 的训练与评测进度：

```bash
RUN="$(cat /data/fsar_workspace/latest_tsfsar_samqam_root.txt)"
tail -n +1 -F "$RUN/shot1.console.log" | grep --line-buffered -E '^SAMQAM_(PHASE|TRAIN |PAIRED|EVALUATION_COMPLETE|PIPELINE_COMPLETE)'
```

`Ctrl+C` 只结束 tail。管理日志位于 `${RUN}.launcher.log`，状态文件 `$RUN/status.json`。

```text
RUN/
  status.json
  shot1.console.log
  shot5.console.log
  shot{1,5}/
    preflight_train/              # 可丢弃检查模块
    preflight_evaluation/smoke/
    train/
      train_identity.json
      training_data_config.json
      train_trace.jsonl
      train_episodes.jsonl
      resume_latest.pth
      module_final.pth            # 只含新模块，不复制整个 E0
      training_complete.json
    evaluation/{smoke,explore,formal}/
      run_identity.json
      effective_config.json
      paired_predictions.jsonl
      completion.json
  formal_analysis/
    accuracy_table.md
    experiment_report.md
    shot{1,5}_statistics.json
    shot{1,5}_prediction_changes.csv
```

`completion.json` 的 `passed=true` 表示完整性/工程检查通过，不表示准确率超过 E0。两个 formal 都应恰有 10,000 条预测。关闭 SAM/QAM 使用 `evaluate --disable-samqam`，需要单独输出目录；该路径直接调用原 DCM，不重写 E0。

## 中断恢复

每 500 个训练更新保存本实验自己的 `resume_latest.pth`。每个训练 Episode 的原始采样与增强由索引固定 RNG 决定，同时保存主进程噪声 RNG、模块 BN 和优化器状态。恢复时会保留 checkpoint 后日志的完整备份，再重建已保存前缀。代码、配置、资源 SHA 或环境不兼容时拒绝恢复。

只有确认旧任务已退出并且该 GPU 空闲后，重新执行该组 `run_gpu.sh`，指定**原来相同的 output**。pipeline 自动识别自己的 checkpoint；已完成阶段校验后跳过。不要再运行 `launch_campus.sh` 来恢复，它创建新实验目录。

```bash
TREE="$(cat /data/fsar_workspace/latest_tsfsar_samqam_tree.txt)"
RUN="$(cat /data/fsar_workspace/latest_tsfsar_samqam_root.txt)"
cd "$TREE"
CHECKPOINT="$(python -c 'import json; print(json.load(open("/data/fsar_workspace/psds_assets/PSDS_READY.json"))["checkpoint_paths"]["1"])')"
# 确认 GPU4 空闲后，在已激活环境下运行：
nohup bash research/sam_qam_experiment/run_gpu.sh \
  --shot 1 --gpu 4 --data-root /data/fsar_workspace/datasets/SSv2-Small \
  --checkpoint "$CHECKPOINT" \
  --manifest "$TREE/research/ps_ds_subspace_experiment/results/shot1/episodes.jsonl.gz" \
  --output "$RUN/shot1" \
  > "$RUN/shot1.resume.$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

预测前缀可按固定 Manifest 继续；损坏或截断的预测 JSON 行会报错，保留证据，不自动删除。单组恢复不会自动重启管理器，两组 formal 完成后可手动执行统计：

```bash
python -u -m research.sam_qam_experiment.analyze \
  --shot1 "$RUN/shot1/evaluation/formal" \
  --shot5 "$RUN/shot5/evaluation/formal" \
  --bootstrap 10000 --output "$RUN/analysis_$(date +%Y%m%d_%H%M%S)"
```

## 本地验证范围

`validation/` 保存 CPU 测试与实际训练数据加载审计。测试包含公式逐项对照、作者固定版本输出对照、1/5-shot E0 还原、分块重计算的梯度和 AdamW 更新、BN 不重复更新、Support 顺序不变性、测试 Query 批次独立性、实际 256×196×8 张量前后向、恢复日志保护及配对统计。

本地没有执行真实编码器推理、GPU 运算或正式训练，没有生成 SAM/QAM 成绩。原始模型源文件、checkpoint、历史结果及运行队列均未修改。新分支只新增本实验目录；旧 PS+DS 工具作为已发布的资源、采样和统计依赖复用。
