# 校园智算平台部署手册

以下命令在项目根目录执行。项目代码不依赖 Codex、VS Code 或原服务器绝对路径。

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
python scripts/calibrate_dto_scale.py --gradients runs/E1_1shot/dto_kd_gradients.csv --output runs/E3_scale.json --config configs/dto_kd/ssv2_small_1shot_E3.yaml
bash scripts/run_campus_experiment.sh --experiment E3 --shot 1 --gpu 0
```

## 9. 梯度诊断

E1/E3 每隔 `GRADIENT_SAMPLE_INTERVAL` 个训练 episode 写入 `dto_kd_gradients.csv`，每隔 `TRACE_INTERVAL` 个 episode 写入 `dto_kd_trace.csv`。字段包括 KL/DCM 范数、余弦、冲突比例、投影、动态权重和日志损失。额外 `autograd.grad` 只在采样间隔执行。

## 10. 评估与汇总

训练期间每 `VAL_FRE_ITER` 个 episode 使用 Meta-Val 保存 `model_best.pth`。方法和权重锁定后，再运行固定 Meta-Test：

```bash
bash scripts/run_campus_experiment.sh --experiment E1 --shot 1 --gpu 0 --mode eval --checkpoint runs/E1_1shot/model_best.pth
```

Meta-Test 结果不能用于选择 DTO 权重或 E2 候选。

## 11. 中断恢复

E1 的 `model_best.pth` 是安全保存的模型权重；从最近 best 继续训练需要将其作为初始化并重新执行同一配置，优化器状态恢复尚未在本机 CUDA 验证。正式结果优先使用完整一次训练的 best checkpoint，避免把未验证的恢复行为混入主结论。

## 12. 错误排查

- CUDA 不可用：检查 `nvidia-smi`、`torch.version.cuda` 和驱动，使用平台匹配的 PyTorch wheel。
- CLIP 缺失或哈希不符：重新导入完整离线归档，禁止换模型。
- 数据路径错误：设置 `TSFSAR_SSV2_ROOT`，确认 `annotations/` 和视频扩展名。
- decord 解码失败：在平台环境安装对应 Linux wheel，并先用 preflight/小样本读取测试。
- OOM：降低平台作业显存占用或保持原 batch/episode 设置；不要修改方法后再比较结果。
- checkpoint 不匹配：确认 shot、配置和 checkpoint 属于同一实验组；四组 checkpoint 不是互换初始化。
