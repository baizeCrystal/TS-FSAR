# CLS-CTM 联合训练：恢复成功 CTM 的训练方式

本轮保留当前 CLS-CTM 的结构，只把冻结 E0、单独学习模块的方式改为原版联合训练。
本目录独立实现，不改原始模型文件、冻结版实现、已有 checkpoint 或历史结果。

## 方法与训练

- Support CLS 的视频时间均值 → 类别均值 → 跨类别均值／总体方差／最近竞争类逐通道平方差。
- MLP `768→256→256`，输出层零初始化，`gate=1+0.1*(256*softmax(logits)-1)`。
- 一组 Episode 共享权重作用于 matching 的 CLS/Patch，再调用原版 TS-DCM。
- 原始未加权 ADC 返回原版全局 CE/KL，原始 CLS 返回 Side CE；TSM 条件仍用原始 CLS。
- 恢复原版 `utils.optimizer.construct_optimizer`：LSN、LSN 全局分类头、Side 文本头、TSM、CLIP Adapter 与 CTM 共同训练；CLIP backbone 冻结。
- CTM 参数通过 `ctm_generator` 名称进入原 generator 组。没有 DTO、动态损失系数、教师 detach、新辅助损失或梯度裁剪。
- 原始全部损失：`VIS_ALPHA*matching_CE + CLS_ALPHA*student_CE + SIDE_CLS_ALPHA*side_CE + KD_ALPHA*(global_KL + teacher_CE)`。
- **1-shot**：四组基础 LR `0.00025`，系数 `1 / 2 / 0.1 / 0.1`。
- **5-shot**：四组基础 LR `0.0003`，系数 `1 / 2 / 0.01 / 0.01`。
- AdamW `eps=1e-6, weight_decay=0.1`，原版余弦调度调用顺序，10,000 个训练 Episode，每次一个完整 Episode，5 Query/类，8 帧。
- AMP 初始 scale=128，沿用此前成功 CTM 的设置；正式训练保留 AMP 自适应回退和被跳过更新的日志。

此前成功的 metric-CTM 是新 CLIP/PCA/随机头初始化后共同训练，而不是从训练好的 E0 best 微调。
因此本轮从官方本地 CLIP 初始化，每种 shot 只生成一次初始模型和文本嵌入库。
E0/E1 严格加载同一库的所有已有参数；新模块构造隔离 RNG。正式训练不继承 smoke 权重。
训练 Episode 使用现有可重放的按索引播种 sampler，原版采样规则／帧数／数据增强不变；
这是为了配对而固定随机流，不声称重现历史 worker 调度和历史成绩的小数。

## 评测与选模

按用户现有要求，前 2499 次不测评；2500、3000、…、10000 次，每次用官方 24 Novel-Test 类、
5-way、1 Query/类、10,000 Episodes，以原版完整三分支乘积融合准确率选 best。
保留原版 **当前 backward 后、当前 optimizer.step 前** 选模顺序，记录此前有效更新数。
原版各 shot 的融合指数完全保留；同时输出 Matching、CLIP、Side、Fusion。
E0/E1 的训练视频／增强随机流和所有测试 Episode Manifest 相同。
保存的 best 会重新进行完整 FP32 原版推理，必须精确重现其选模准确率后才标记完成。
这是按要求采用 Test 选模，最终 Manifest 与选模相同，不是独立确认集。
配对 bootstrap CI 仅描述所选 checkpoint 下 Episode 抽样不确定性。

## A10 内存与验收

训练使用 PyTorch `save_on_cpu(pin_memory=True)` 保存反向传播激活；损失和梯度路径不 detach。
不改变 Episode 大小，也不把时间帧切开。会增加 CPU 内存和 CPU/GPU 传输开销。
后台队列先执行：

1. 本地 CPU 数值测试，资源 SHA256 验证；
2. 官方 CLIP/PCA 新初始化库；
3. CUDA 小模型的原地保存／CPU 暂存在 AMP 下梯度和连续 Adam 更新精确比较；
4. 真实训练 Episode 的 E0/E1 恒等初始化损失、原辅助输入和已有梯度对比；
5. 原始所有训练组和新 CTM 都发生有效更新；
6. E0、E1 各四个有效 GPU 更新，含 AMP backoff，随后各三个真实 Test Episodes 完整融合检查。

双方 GPU 验收通过才启动正式训练。原服务器没有执行 GPU 验收或 GPU 训练，
测试数值通过也不预先保证校园 A10 的实际峰值；峰值在预检及训练日志记录。

## 校园服务器：先只用 GPU4 跑 1-shot

```bash
(
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd
export PATH="$HOME/.local/gh-cli/bin:$HOME/.local/bin:$PATH"
cd /data/fsar_workspace/TS-FSAR-CLSCTM
test "$(git branch --show-current)" = research/cls-ctm-discriminative
git pull --ff-only
bash research/cls_ctm_joint_experiment/launch_campus.sh --shots 1 --gpus 4
)
```

每种 shot 独占一张 GPU。默认只运行 1-shot/GPU4，GPU5 和其他卡没有动作。
每种 shot 先完成两组预检，再 **E1 联合训练 → E0 同预算对照 → 配对统计**，
其中每组都训练 10,000 Episodes。E0 对照是新训练的同初始化基线，不是冻结版历史 60.218%。
若两卡确实空闲，可改为 `--shots 1,5 --gpus 4,5`，两种 shot 并行。
指定 GPU 有进程或异常 ECC 时拒绝启动，不停止已有任务。

整个准备／验收／训练／测评由 nohup 后台管理，脚本立即打印管理日志的绝对路径。
所有新 checkpoint、输出、临时文件置于 `/data/fsar_workspace/`，资源复用 `psds_assets`。
Github 资源需要之前已经登录的 `gh`，不新建仓库、不上传数据。

准备／验收日志：

```bash
RUN="$(cat /data/fsar_workspace/latest_tsfsar_clsctm_joint_root.txt)"
tail -n 40 -F "$RUN/shot1.console.log"
```

正式 E1 日志（预检结束后才出现）：

```bash
RUN="$(cat /data/fsar_workspace/latest_tsfsar_clsctm_joint_root.txt)"
tail -n 40 -F "$RUN/shot1/E1.console.log"
```

只看测评：

```bash
tail -n +1 -F "$RUN/shot1/E1.console.log" | grep --line-buffered '^TEST_SELECTION '
```

双方完成后的报告：`$RUN/shot1/analysis/experiment_report.md`。
统计：`$RUN/shot1/analysis/statistics.json`。
训练记录：`$RUN/shot1/{E0,E1}/train_trace.jsonl` 与 `train_episodes.jsonl`。
模型：`$RUN/shot1/{E0,E1}/model_best.pth`、`resume_latest.pth`。

## 中断与续训

旧输出不会由新队列覆盖。某个 shot 因非数值错误中断后，先确认原进程退出、GPU 空闲，
可用同一 checkout 和相同 shot/data/output 再执行 `run_gpu.sh`；它自动恢复已有 latest：

```bash
bash research/cls_ctm_joint_experiment/run_gpu.sh --shot 1 --gpu 4 \
  --data-root /data/fsar_workspace/datasets/SSv2-Small --output "$RUN/shot1"
```

它会保留 checkpoint 之后的训练日志为带时间戳备份；若测试预测已写入但未纳入保存的 latest，
拒绝覆盖这些现场文件，需要先检查和归档后再恢复。
代码哈希、初始化、Manifest 或配置不一致时拒绝恢复。

## 本地数值验证

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s research/cls_ctm_joint_experiment/tests -v
```

数值用小型合成编码器调用真实 TS-FSAR、TS-DCM、Alpha_DC、DistillKL 和官方优化器，
不下载模型、不安装依赖，不把合成数据的准确率当作实验结果。
