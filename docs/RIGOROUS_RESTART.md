# 校园实验严谨重启：protocol v2

旧队列 `runs/campus/a10_seed0_20261107_103208` 停止后完整保留，不继续训练、不覆盖、不作为 v2 初始化。原服务器只开发和 CPU 测试；下面涉及 CUDA 的命令仅在校园已分配 GPU 上执行。GPU 7 曾有不可纠正 ECC，在管理员处理前排除。

## 冻结的实验定义

所有组使用本仓库保留的发布配置合并 `configs/base.yaml`：5-way、8 帧、训练每类 5 query、评估每类 1 query，episode batch=1、累计=1，10000 个训练 episode，每 500 个 episode 在固定 10000 个 Meta-Val episode 选 best，原 AdamW/eps/decay 和 cosine_lr。保留发布代码在 optimizer step **之前**验证/保存 best 的时序；日志明确记录选点时已完成的更新数。1-shot LR=0.00025、5-shot LR=0.0003。原三分支融合、训练/最终 Test 文本、辅助 CE、教师 Adapter 梯度不变。

v2 重新组织随机数：原采样器/增强使用 split/shot/seed/episode 独立种子；不更改视频采样规则、帧数或图像增强。所有新组包括 E0 都使用 v2，不能将历史不同流结果强行配对。实际视频、帧、Support/Query 顺序、标签、增强后输入 SHA256 写入 ledger；正式第 0 个 episode 必须与 CUDA 验收的输入完全一致。

历史 60.774%/70.696% 是 Test 周期 best，v2 用 12 类 Meta-Val 选 best。历史数值只作注明协议的描述性参考；方法收益以新 E1/E2 减新 E0 计算。不能把旧/新差值归因于单项修复。

| 组 | 1-shot | 5-shot |
|---|---|---|
| E0 | 原损失，KL=0.1 | 原损失，KL=0.01 |
| E1 | Eq.18 直接更新概率再 softmax | 相同 |
| E2 低/高权重 | KL=0.05 / 0.2 | KL=0.005 / 0.02 |
| E2 中心 | 复用同目标 E0 | 复用同目标 E0 |

E1 预先固定 eta=0.01、初始 pi=0.5/0.5、loss clamp epsilon=1e-8；这是按论文 Eq.18 公式的 **FSAR 适配版**，仍保留原可训练教师及三个辅助 CE，并非作者图像分类系统的逐行复现，也没有在全图上使用双梯度闭式解。不能直接声称理论上消除了冲突。

当前八个 seed0 训练任务足以比较该 seed 内方法，不能估计训练种子方差。正式稳健结论预留 seed=0/1/2 的同协议重复，各 seed 单独做初始化与验收；未运行的种子不得补造。E2 只在 Meta-Val 选择，不能根据 Test 换范围。本次将静态候选固定为原 KL 的半倍/两倍，来自原配置，不来自 Test 搜索；旧 legacy 网格仍保留在历史文件中。

## 1. 停止与保存旧队列

在校园项目目录执行；先核对启动器身份，发送 SIGTERM 后由旧启动器处理自己的进程组，不使用 `pkill python`。

```bash
python - <<'PY'
import os, signal
from pathlib import Path
root = Path('runs/campus/a10_seed0_20261107_103208').resolve()
pid = int((root / 'launcher.pid').read_text().strip())
proc = Path('/proc') / str(pid)
if not proc.exists():
    print('Launcher already exited; inspect nvidia-smi for remaining jobs.')
else:
    args = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    cwd = proc.joinpath('cwd').resolve()
    if not (any(Path(a).name == 'launch_campus_suite.py' for a in args)
            and '--output-root' in args
            and (cwd / args[args.index('--output-root') + 1]).resolve() == root):
        raise SystemExit('PID identity mismatch; no signal sent')
    os.kill(pid, signal.SIGTERM)
    print('Sent SIGTERM to this launcher:', pid)
PY
nvidia-smi
```

信号已发送不等于已验证所有子进程退出。若旧启动器消失而 GPU 仍占用，根据原 `status.json` 和 `/proc/PID/cmdline` 定位遗留进程，不能按过期 PID 盲杀，也不要重置其他正在使用的 GPU。

## 2. 更新与 CPU 检查

```bash
git pull --ff-only origin research/dto-kd-fsar-gradient-coordination
export TSFSAR_SSV2_ROOT=/data/fsar_workspace/datasets/SSv2-Small
export TSFSAR_RUNTIME_TMPDIR=/tmp
python scripts/audit_campus_protocol.py
python scripts/check_ssv2_files.py
python scripts/verify_weights.py
CUDA_VISIBLE_DEVICES='' python -m unittest \
  tests.test_dto_kd_cpu tests.test_campus_execution_cpu \
  tests.test_dto_kd_memory_cpu tests.test_campus_protocol_cpu \
  tests.test_rigorous_protocol_cpu tests.test_campus_amp_acceptance_cpu -v
```

文件检查覆盖全部 canonical 视频的存在、大小和读权限，不是全视频解码。禁止更换 CLIP 文件；加载形状不匹配会报错，取消随机替换 fallback。

## 3. 校园 CUDA 验收：先做这一阶段

确认旧进程全部退出后，只占两张健康卡；这是完整真实 episode 的验收与短 smoke，不是正式训练。每种 shot 先让原 E0 在固定输入、完整初始权重和相同 RNG 下从默认 GradScaler scale=65536 开始，最多 17 次尝试（通常按半衰减，到 scale=1），直到出现一次有限梯度且实际改变参数的 AdamW 更新。跳过的尝试必须不改变参数和优化器状态，全部写入 `amp_attempts.jsonl`。关闭适配/静态原系数的候选各只尝试一次，必须在原 E0 的同一成功 scale 下通过 loss、梯度、参数更新对比；不能把两组都未更新当作等价性通过。

然后做 E0/E1 各固定 32 个训练 episode 的 smoke（第 2 步梯度诊断）和各 2 个 Val episode。smoke 与正式训练均使用原默认 GradScaler，不使用校准后的较低初始 scale；记录初始溢出跳步，每组至少需有 3 次有效参数更新。验收校准不会传递模型、优化器或 scaler 的训练后状态给正式训练。超过重试上限、非有限前向 loss、候选溢出、更新数不足、OOM、数值差异或非零 ECC 均拒绝签发通过报告。首次 AMP 跳步本身不再直接判失败。

```bash
ACCEPT_ROOT="runs/campus/acceptance_v2_seed0_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ACCEPT_ROOT"
echo "$ACCEPT_ROOT" > runs/campus/latest_acceptance_root.txt

nohup bash scripts/run_campus_acceptance.sh \
  --shot 1 --seed 0 --gpu 0 --output "$ACCEPT_ROOT/shot1" \
  > "$ACCEPT_ROOT/shot1.launcher.log" 2>&1 < /dev/null &
echo "$!" > "$ACCEPT_ROOT/shot1.pid"

nohup bash scripts/run_campus_acceptance.sh \
  --shot 5 --seed 0 --gpu 1 --output "$ACCEPT_ROOT/shot5" \
  > "$ACCEPT_ROOT/shot5.launcher.log" 2>&1 < /dev/null &
echo "$!" > "$ACCEPT_ROOT/shot5.pid"

tail -f "$ACCEPT_ROOT/shot1.console.log" "$ACCEPT_ROOT/shot5.console.log"
```

如果 shell 变量丢失：`ACCEPT_ROOT="$(cat runs/campus/latest_acceptance_root.txt)"`。`acceptance.json` 存在且 passed=true 才表示该次检查通过；仅有日志、初始化文件、失败报告或 shell 后台 PID 都不代表成功。查看生成的最大误差与 tolerance、AMP backoff 记录、实际有效更新数，不能将容差通过描述为位级相同。32 个 episode 的 smoke 也不是全程训练稳定性证明。早期因 first-step AMP overflow 失败的目录完整保留，更新后使用新的验收输出目录。

报告绑定代码指纹、shot/seed、PyTorch、GPU 型号与完整初始化 SHA256。新的全部实验严格加载同一个经过验证的初始状态，避免每个任务重复 CPU PCA，恢复同一初始 Dropout 随机流。不会从历史训练 checkpoint 开始。

## 4. 正式训练：两个验收通过后再执行

先只启动 E0 两组也可，将 `--experiments E0,E1,E2` 改成 `--experiments E0`。默认队列是 v2，缺验收报告会拒绝启动；历史运行只能显式选择 `--protocol legacy`。

```bash
ACCEPT_ROOT="$(cat runs/campus/latest_acceptance_root.txt)"
RUN_ROOT="runs/campus/rigorous_v2_seed0_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

nohup python -u scripts/launch_campus_suite.py \
  --protocol v2 --experiments E0,E1,E2 \
  --gpus 0,1,2,3,4,5,6 --seed 0 \
  --acceptance-1shot "$ACCEPT_ROOT/shot1/acceptance.json" \
  --acceptance-5shot "$ACCEPT_ROOT/shot5/acceptance.json" \
  --activation-offload diagnostics --launch-delay 20 \
  --output-root "$RUN_ROOT" \
  > "$RUN_ROOT/launcher.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_ROOT/launcher.pid"
```

共 8 个任务，最多 7 个同时运行；每进程单卡，保持 episode batch=1。GPU 7 不使用。所有实验使用同一 CPU activation 保存策略，只改变存储，不改变损失/采样。若校园验收失败，先解决失败，不直接启动队列。正式训练的 source fingerprint、环境、基线配方审计和第 0 个真实输入会再次核对。

## 5. 日志与完整恢复

每个 job 保存 console、frozen_config、protocol_v2、train_episodes、val_episodes、train_trace、validation、gradient_groups、predictions、model_best.pth 和 resume_latest.pth。梯度在最初 500 步每 25 步、以后每 500 步采样，记录共享 LSN、LSN 文本头、全局头、教师 Adapter、TSM generator、全体可训练参数的原始/加权范数、余弦和投影，以及三项 CE / KL 加三项 CE 相对任务方向。零范数的非共享分组不解释为消除冲突。

原配方无梯度裁剪，记录 unscaled 总梯度和实际参数更新范数；AMP 跳过更新会记录并回滚该步 DTO 控制器更新。各组训练 episode 预算相同，是否有不同数量的有效 AMP 更新必须另行报告。

完整 checkpoint 每次 Val 后及训练结束保存，包含模型、优化器、scaler、控制器、Python/NumPy/CPU/CUDA RNG、已完成 episode、best 状态和原 cosine 定义。恢复会回放尚未提交的 episode；代码/配置/已提交 ledger 不一致会拒绝。没有写入 checkpoint 的中断尾部不是已完成训练。仅 CPU 合成恢复已经测试；真实模型 CUDA 恢复的数值一致性尚不能声称已验证。

```bash
bash scripts/run_campus_experiment.sh \
  --protocol v2 --experiment E1 --shot 1 --gpu 0 --seed 0 \
  --acceptance-report "$ACCEPT_ROOT/shot1/acceptance.json" \
  --resume "$RUN_ROOT/E1_1shot/resume_latest.pth" \
  --output "$RUN_ROOT/E1_1shot"
```

恢复前必须确认该 job 已退出；锁阻止同一输出/GPU 并发。模型 best 不是完整恢复文件。不要重新初始化优化器后把训练拼成原协议。

## 6. E3 与最终 Test

E3 使用 E1 最初 500 个 Meta-Train episode 中 20 个共享 LSN 采样点的实际加权 KL/DCM 范数比；只匹配相对贡献，不声称匹配绝对尺度或排除了全部因果混杂。新工具拒绝不足 20 个点、错误来源和重复点，读原模板并生成新配置：

```bash
python scripts/calibrate_rigorous_scale.py \
  --gradients "$RUN_ROOT/E1_1shot/gradient_groups.jsonl" \
  --config configs/dto_kd/ssv2_small_1shot_E3.yaml \
  --output-config "$RUN_ROOT/E3_1shot_frozen.yaml" \
  --output-report "$RUN_ROOT/E3_1shot_calibration.json"
```

E3 训练使用 `--experiment E3 --config` 指向新配置，其余 v2 验收、seed、shot 和输出要求相同；5-shot 用对应模板/报告。E3 不自动排队。

方法、E2 系数和全部设置在 Meta-Val 锁定后才运行最终 Test，始终使用新的评估输出目录：

```bash
bash scripts/run_campus_experiment.sh \
  --protocol v2 --experiment E1 --shot 1 --gpu 0 --seed 0 --mode eval \
  --acceptance-report "$ACCEPT_ROOT/shot1/acceptance.json" \
  --checkpoint "$RUN_ROOT/E1_1shot/model_best.pth" \
  --output "$RUN_ROOT/E1_1shot/final_test"
```

`test_result.json` 给出四分支与该次预测文件。对 E0 使用相同 shot/seed 的最终评估，然后 CPU 比较；路径从两份 test_result.json 的 predictions 字段读取：

```bash
python scripts/compare_rigorous_results.py \
  --reference /实际/E0的test预测.jsonl \
  --candidate /实际/E1的test预测.jsonl \
  --bootstrap 5000 --seed 0 --output paired_E1_minus_E0.json
```

只有 Episode ID、完整输入哈希、Query 标签完全一致才配对。Bootstrap 以 episode 为单位，输出四分支差值、未作多重比较校正的 95% CI、共同正确/错误、纠正/新增错误。单个训练 seed 的 CI 不代替多种子结论，也不能据 Test 选择新超参数或融合权重。

## 验证边界

原服务器验证记录（2026-11-07，AMP 验收修复后）：44 项测试中 38 项 CPU 测试通过、6 项 CUDA 测试跳过；14 份源实验配置的基础配方审计通过。CPU 集成覆盖实际采样器的 worker 一致性、TS_DCM/GLAC 损失路径、控制器公式和完整状态恢复；新增 AMP 模拟测试覆盖固定 RNG 重放、跳步无参数/优化器变化、同 scale 一次有效更新及持续溢出拒绝。视觉编码器由合成特征替代、AMP backoff 在 CPU 由模拟 scaler 注入，不能作为真实完整模型或 CUDA AMP 验收。

原服务器只执行 CPU 合成测试，真实模型 CUDA 验收需要校园执行并提供产物。v2 原始数据/网络/损失/官方基础配方保持不变，但 RNG 组织、选模协议和 E1 直接权重更新均与旧校园批次有差异；旧/新对照必须如实标注。即便所有工程验收通过，也不能保证准确率提升；有效的研究结果可能是不提升或负结果。
