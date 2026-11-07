# 校园智算平台部署手册

以下命令在项目根目录执行。项目代码不依赖 Codex、VS Code 或原服务器绝对路径。

**正式重启请优先执行 [protocol v2 手册](RIGOROUS_RESTART.md)。** 当前校园 runner/queue 默认 v2，要求通过校园完整模型 CUDA 验收后才能正式训练。本文后续早期示例保留用于追溯，缺验收参数的正式命令会被拒绝；旧逻辑只能显式 `--protocol legacy`。历史审计的缺失项及 v2 已补项目见新手册，不能把 CPU 测试当作校园 CUDA 验收已通过。

2026-11-07 正式实验审计发现：`62d6f01` 及更早校园 DTO 配置的所有 5-shot 组误用 `0.00025` 学习率；原始发布配置及本地 70.696% 复现使用 `0.0003`。本次已修正未来运行的六份 5-shot 配置，既有输出仍保留为旧学习率探索实验，不能中途改 LR 后拼接成正式结果。E1 还是 softmax-logit 控制器适配版；完整研究验收尚有未完成项。开始新的正式训练前请阅读 [审计及证据边界](audits/CAMPUS_PROTOCOL_AUDIT_20261107.md)。

只读 CPU 配置核对：

```bash
python scripts/audit_campus_protocol.py
# 审计已经生成的实际运行配置（替换为实际目录）：
python scripts/audit_campus_protocol.py --run-root runs/campus/a10_seed0_20261107_103208
```

退出码 0 只表示所检查的配置与划分通过，不认证 CUDA 数值一致性、算法严格复现或最终效果。

## 1. 获取项目

```bash
git clone --branch research/dto-kd-fsar-gradient-coordination https://github.com/baizeCrystal/TS-FSAR.git
cd TS-FSAR
```

若仓库为私有，先用校园平台的 GitHub HTTPS 登录方式完成认证。数据集视频和大型权重不在 Git 历史中。

## 2. 创建环境

推荐 Python 3.8.20、PyTorch 2.2.2、torchvision 0.17.2、Transformers 4.46.3、decord 0.6.0。当前 A10 每卡约 23028 MB；配置固定为每个 episode 一个 task（`TRAIN.BATCH_SIZE=1`、`BATCH_SIZE_PER_TASK=1`），这是原协议要求，也给 DTO 采样梯度保留显存余量。不要把这个 episode batch 提高到 2，否则需要重新锁定训练协议。CUDA wheel 必须与校园驱动兼容；不要修改系统驱动。无 sudo 时使用平台已有 CUDA 容器或用户目录 Conda：

```bash
bash scripts/setup_campus.sh
conda activate tsfsar-dto-kd
```

脚本默认安装 CUDA 12.1 wheel；若平台驱动只支持其他构建，先设置 `TSFSAR_TORCH_INDEX_URL` 或使用平台预装 PyTorch，再执行 requirements 安装。CPU 只能做 preflight 和单元测试，不能完成正式视频训练。

## 3. 准备预训练权重

```bash
python scripts/download_pretrained.py
python scripts/verify_weights.py
```

不能联网时使用离线资源目录：

```bash
bash scripts/import_offline_assets.sh /path/to/offline_assets
python scripts/verify_weights.py
```

归档中的目录必须是 `model/vit-b-16/`。权重哈希见 `assets/CLIP_VIT_B16_SHA256SUMS.json`。

## 4. 配置数据集

SSv2-Small 应解压为：

```
/data/fsar_workspace/datasets/SSv2-Small/*.mp4
/data/fsar_workspace/datasets/SSv2-Small/annotations/{train,val,test}_few_shot.txt
```

可以通过 `TSFSAR_SSV2_ROOT` 指向其他目录；不要改 Python 代码。数据集视频不由本仓库重新分发，必须按原始官方许可获取。先检查：

```bash
TSFSAR_SSV2_ROOT=/data/fsar_workspace/datasets/SSv2-Small bash scripts/campus_preflight.sh
```

## 5. 运行 Preflight

```bash
bash scripts/campus_preflight.sh
python -m unittest tests.test_dto_kd_cpu -v
```

它会检查 Python、Torch/CUDA、Transformers、decord、CLIP 文件和四个划分元数据。

## 6–8. 运行 E0/E1/E2/E3

E0 原始策略：

```bash
bash scripts/run_campus_experiment.sh --experiment E0 --shot 1 --gpu 0
bash scripts/run_campus_experiment.sh --experiment E0 --shot 5 --gpu 0
```

E1 DTO-KD：

```bash
bash scripts/run_campus_experiment.sh --experiment E1 --shot 1 --gpu 0
bash scripts/run_campus_experiment.sh --experiment E1 --shot 5 --gpu 0
```

E2 静态 KL 候选必须逐个运行并只用 Meta-Val 选择：

```bash
python scripts/prepare_run_config.py --config configs/dto_kd/ssv2_small_1shot_E2_static_kl_0.1.yaml --output runs/E2_1shot/config.yaml --output-dir runs/E2_1shot
CUDA_VISIBLE_DEVICES=0 python -m experiments.dto_kd.entry --cfg runs/E2_1shot/config.yaml
```

把 `0.05`、`0.1`、`0.2` 三个候选分别运行。E3 先用 E1 的 Meta-Train 梯度诊断校准：

```bash
python scripts/calibrate_dto_scale.py --gradients runs/campus/E1_1shot_seed0/dto_kd_gradients.csv --output runs/E3_scale.json --config configs/dto_kd/ssv2_small_1shot_E3.yaml
bash scripts/run_campus_experiment.sh --experiment E3 --shot 1 --gpu 0
```

## 9. 梯度诊断

E1/E3 每隔 `GRADIENT_SAMPLE_INTERVAL` 个训练 episode 写入 `dto_kd_gradients.csv`，每隔 `TRACE_INTERVAL` 个 episode 写入 `dto_kd_trace.csv`。字段包括 KL/DCM 范数、余弦、冲突比例、投影、动态权重和日志损失。额外 `autograd.grad` 只在采样间隔执行。

## 10. 评估与汇总

校园启动脚本通过生成独立划分副本，将训练期评估固定为 12 类 Meta-Val（不使用 24 类 Test）。原入口仍将评估 loader 称为 `test`；`meta_val_annotations/protocol.json` 明确记录实际来源。Val 文本为已有类别名描述，所有组固定一致，训练与最终 Test 文本保持原文件。历史 Test 选点 checkpoint 只作为参考，不与这轮 Val 选点结果当作完全相同协议。训练期间每 `VAL_FRE_ITER` 个 episode 使用 Meta-Val 保存 `model_best.pth`。方法和权重锁定后，再运行固定 Meta-Test：

```bash
bash scripts/run_campus_experiment.sh --experiment E1 --shot 1 --gpu 0 --mode eval --checkpoint runs/campus/E1_1shot_seed0/model_best.pth
```

Meta-Test 结果不能用于选择 DTO 权重或 E2 候选。

## 11. 中断恢复

当前只保存 `model_best.pth` 模型权重，没有完整优化器、AMP、RNG、已完成 episode 和 DTO 控制器的训练恢复入口。不能把读取 best 权重描述成原状态的继续训练，也不能使用 `--checkpoint` 恢复训练（该参数只用于评估）。中断后保留原记录；重跑使用新目录和相同初始化从头开始。配置里的 `AUTO_RESUME` 不代表已经实现恢复。

## 12. 错误排查

- CUDA 不可用：检查 `nvidia-smi`、`torch.version.cuda` 和驱动，使用平台匹配的 PyTorch wheel。
- CLIP 缺失或哈希不符：重新导入完整离线归档，禁止换模型。
- 数据路径错误：设置 `TSFSAR_SSV2_ROOT`，确认 `annotations/` 和视频扩展名。
- decord 解码失败：在平台环境安装对应 Linux wheel，并先用 preflight/小样本读取测试。
- OOM：降低平台作业显存占用或保持原 batch/episode 设置；不要修改方法后再比较结果。
- checkpoint 不匹配：确认 shot、配置和 checkpoint 属于同一实验组；四组 checkpoint 不是互换初始化。


## 13. 校园运行修复、完整日志与八卡并行

先 `git pull --ff-only origin research/dto-kd-fsar-gradient-coordination`。
本次修复 E1/E2/E3 support/query 标签留在 CPU 导致的 `index_select`
设备冲突。`tests/test_campus_execution_cpu.py` 用 CPU 搬运探针执行三个合成
优化步骤，检查标签和图像经过相同 CUDA 搬运路径；这不是实际 CUDA 验证。

`the N th side layer is initialized...` 来自逐层 CPU PCA 初始化，不代表
第 N 个训练 episode。每个进程默认限制 OpenMP/MKL/OpenBLAS 为 2 线程，
可用 `TSFSAR_CPU_THREADS` 配置。更换线程数的实际加速幅度需要校园端测量，
固定同一轮实验组的线程设置。DataLoader 仍为原来的 4 个 worker，8 个任务
同时运行时还需要足够的 CPU、主存、共享内存和数据盘吞吐。不要为并行把
`NUM_GPUS` 改成 8：每个独立进程始终 `NUM_GPUS=1`、episode batch=1。

环境及 pip 安装可用 `/data` 的 TMPDIR；训练时脚本会改用本地 `/tmp`
创建每进程临时目录，避免 NFS 多进程 socket 清理时出现 `.nfs... resource busy`。
本地 `/tmp` 仍须有可用空间；如另有本地临时盘，设置
`TSFSAR_RUNTIME_TMPDIR=/path/to/local/temp`。不要指向 NFS。

先在已分配的 GPU 上执行短检查（每次 3 个训练/2 个验证 episode，
第 2 步诊断；覆盖普通步、诊断步、Adam 状态建立后的普通步。仅用于查错
和显存检查，不是正式研究结果）：

```bash
python -m unittest tests.test_dto_kd_cpu tests.test_campus_execution_cpu -v
# 先用小张量检查控制器跨设备更新、AMP 和状态恢复；不加载 CLIP/PCA/视频。
TSFSAR_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 python -m unittest tests.test_dto_kd_cuda -v
bash scripts/run_campus_experiment.sh --experiment E1 --shot 1 --gpu 0 --smoke --output runs/campus/smoke/E1_1shot
bash scripts/run_campus_experiment.sh --experiment E1 --shot 5 --gpu 0 --smoke --output runs/campus/smoke/E1_5shot
```

检查日志中训练、验证和梯度诊断均完成，且无 OOM，再启动正式队列。
以下以平台已分配 0–7 全部八张卡为前提；脚本不会更改平台分配或抢占
已有进程。若其中一张已有任务，传入其他 GPU，并用 `--skip E1_1shot`
等排除正在运行的同名实验。

```bash
python scripts/launch_campus_suite.py --gpus 0,1,2,3,4,5,6,7 --dry-run
mkdir -p runs/campus
nohup python -u scripts/launch_campus_suite.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-root runs/campus/parallel_seed0 \
  > runs/campus/parallel_seed0.launcher.log 2>&1 < /dev/null &
echo "queue PID: $!"
```

队列包括 E0、E1、E2 三个 KL 候选的 1-shot/5-shot，总共 10 个任务，
同时最多 8 个，一张卡一个独立实验。默认每隔 20 秒启动一个任务，
完成后空出的 GPU 领取下一个任务。任务失败不自动重试，未启动的任务
会取消，已经运行的其他任务继续。检查报错后用新的 `--output-root`
重跑所需范围；不自动覆盖先前队列记录。

E3 不会自动启动：它依赖 E1 的 Meta-Train 梯度校准。E2 只允许根据
Meta-Val 选择候选，队列不读取或比较最终 Test 准确率。

```bash
# 启动、完成、失败、PID 和 GPU 分配
 tail -f runs/campus/parallel_seed0.launcher.log
# 结构化状态（queued/running/completed/failed/cancelled_after_failure）
 python -m json.tool runs/campus/parallel_seed0/status.json
# 单实验初始化、训练、Python 报错的完整终端记录
 tail -f runs/campus/parallel_seed0/E1_1shot/console.log
# 原训练统计；每 100 个 episode 写训练进度
 tail -f runs/campus/parallel_seed0/E1_1shot/training_log.log
# GPU 状态
 nvidia-smi
```

每个实验独立保存 `config.yaml`、`console.log`、`runner.log`、
`training_log.log`、`best_result.json`、`model_best.pth`；适用组另有
`dto_kd_trace.csv` 和 `dto_kd_gradients.csv`。`runner.log` 还包含
打开 console 日志之前的启动错误。评估日志写入 `eval_console.log`，
评估配置写入 `config_eval.yaml`，避免覆盖训练配置。

如果需要终止此队列，使用它输出的 PID 执行 `kill -TERM PID`；队列仅向
自己启动的进程组发送终止信号，未实现完整优化器断点恢复，不要将终止
后从头启动称为恢复训练。`nohup` 只能防止终端断开导致的 SIGHUP，
平台作业到期、容器销毁或资源回收仍会停止任务。

### 控制器 CPU / CUDA 边界

`DTOController` 不是 `nn.Module`，其两个 FP64 logit 留在 CPU，不会被
`model.cuda()` 搬运。模型的加权 log-loss 在损失所在 GPU 上用 FP32
计算，保持反向图。控制器更新只将两个 detached log-loss 标量搬到
控制器设备后做点积，防止 `torch.dot` 混用 CPU 权重和 CUDA 损失。
状态恢复也将控制器 logit 固定回 CPU，包括从 `map_location='cuda'`
读取的 checkpoint。这个修复不改变权重更新公式或优化目标。

原服务器仅执行 CPU 测试；`tests.test_dto_kd_cuda` 默认跳过，仅在校园
已分配 GPU 上设置 `TSFSAR_RUN_CUDA_TESTS=1` 后执行。请先确认这些
小张量 CUDA 测试全部通过，再运行较慢的模型短检查。


### A10 诊断显存峰值与 CPU 暂存

降低 `GRADIENT_SAMPLE_INTERVAL` 的采样频率不会降低采样时的显存峰值。
`loss_gradients` 现在将每个目标的诊断梯度立即转到 CPU，释放 GPU 上
该目标的梯度后才计算下一个目标，不改参数 `.grad`。方向、范数和投影
统计在 CPU 完成。诊断复用训练时已计算的任务 CE，不另建一份 CE 图；
正常 backward 后清除 episode 缓存引用，下一步不保留上一步输入。

默认 `--activation-offload diagnostics` 在采样 episode 中启用
`torch.autograd.graph.save_on_cpu(pin_memory=False)`：将反向保存的张量
暂存 CPU，反向时恢复到原设备。没有重新执行 forward，没有更换损失、
帧数、Query 数、episode 或优化器；代价是采样时增加 CPU 主存占用和
CPU/GPU 传输。完整模型的实际 GPU/CPU 峰值仍须校园端短检查测量。

如果短检查的第 1 或第 3 个普通训练步仍 OOM，可统一改成所有 episode
暂存 CPU。E0 仍保留原模型和训练循环，仅改变保存张量的存储位置：

```bash
bash scripts/run_campus_experiment.sh --experiment E1 --shot 5 --gpu 0 \
  --smoke --activation-offload all --output runs/campus/smoke/all_offload/E1_5shot
# 确认主存余量和两种 shot 的短检查均通过后，再启动多卡队列。
# 所有组统一使用同一策略，避免把其速度变化误归为算法开销。
python scripts/launch_campus_suite.py --gpus 0,1,2,3,4,5,6,7 \
  --activation-offload all --output-root runs/campus/parallel_all_offload --dry-run
```

`all` 会增加每个任务的 CPU 主存和 PCIe 传输开销。启动八个任务前检查
`free -h`，以及容器的 `/sys/fs/cgroup/memory.max`（cgroup v2）或
`/sys/fs/cgroup/memory/memory.limit_in_bytes`（v1）。不能仅根据宿主机
总内存决定并发数；主存不足时减少同时运行的任务，而非更改实验 episode。
`--activation-offload off` 用于明确关闭暂存的对照诊断，不是 A10 推荐设置。
