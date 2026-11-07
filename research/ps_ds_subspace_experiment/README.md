> **从 GitHub 使用：先看 [GITHUB_QUICKSTART.md](GITHUB_QUICKSTART.md)。不需要网页上传代码包，也不需要新建仓库。**

# 冻结 TS-FSAR：E0 与唯一 PS→DS 方案

实现与数值检查已完成；**尚未进行 GPU 模型推理，200/10000 Episode 准确率待测**。不要将历史 60.77/70.70 填入本次 E0 单元格。

方法、论文对应关系、明确的迁移选择与限制见 [METHOD.md](METHOD.md)。工程状态和待测表见 [experiment_report.md](experiment_report.md)。所有新文件均在本目录；未改原模型源码、原 checkpoint 或任何训练队列。

## 现成资产

- `subspace.py`：Support 视频 SVD、PS、DS、完整 Cayley 优化。
- `matching.py`：类别条件 256 通道投影，原 α-D、原 TSM，关闭开关精确走 E0。
- `configs/locked.json`：固定 r/步数/lr、官方配置/划分/prompt/函数指纹、CLIP 校验、各 shot 的 E0 checkpoint SHA。
- `results/shot1/episodes.jsonl.gz`、`results/shot5/episodes.jsonl.gz`：已实际生成的 10000 Episode 清单。不要重新抽样。
- 每种 shot 的 `manifest_meta.json`、`video_metadata.json`、`sampler_pixel_audit.json` 必须随 Manifest 一起复制。
- `evaluate.py`：原模型 forward 对照、配对预测、冻结状态检查、日志、断点续跑。
- `analyze.py`：10000 次 Episode bootstrap、纠正/新增错误、Hard Negative、耗时、正交/秩与尺度统计。
- `run_gpu.sh`：按 smoke → 200 探索 → 10000 正式执行一套 shot；没有任何训练入口。
- `psds_evaluation_code.tar.gz`：独立可运行的轻量代码快照，包含上述清单及必需原版 runtime，**不含数据、环境二进制或权重**。

## 校园端准备

推荐复用已有 `/data/fsar_workspace/envs/tsfsar-dto-kd` 环境（Python 3.8、torch 2.2.2+cu121、torchvision 0.17.2、decord 0.6.0 等）；依赖版本列于代码包根目录 `requirements-campus.txt`。本方法只用现有 PyTorch/NumPy/PyYAML，不需要新安装大型依赖。环境不匹配时先检查，不要在已运行的训练环境里直接升级。

把本目录的 `psds_evaluation_code.tar.gz` 及其 `.sha256` 上传校园端。小型代码包可用网页上传。请解压到**新目录**，不要覆盖现有训练 worktree：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd

# 先 cd 到实际上传了这两个文件的目录。
sha256sum -c psds_evaluation_code.tar.gz.sha256
mkdir -p /data/fsar_workspace/psds_eval
test ! -e /data/fsar_workspace/psds_eval/TS-FSAR-PSDS-eval
tar -xzf psds_evaluation_code.tar.gz -C /data/fsar_workspace/psds_eval
cd /data/fsar_workspace/psds_eval/TS-FSAR-PSDS-eval

python - <<'PY'
import json,hashlib
from pathlib import Path
for name,want in json.loads(Path('BUNDLE_FILES_SHA256.json').read_text())['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want,name
print('CODE_AND_MANIFEST_VERIFIED')
PY

# 只给新评测目录创建 CLIP 链接，指向此前已校验的文件。
# 若原模型文件在其他 worktree，替换下面 SOURCE_TREE 的值。
SOURCE_TREE=/data/fsar_workspace/datasets/TS-FSAR-blocking/TS-FSAR
test -f "$SOURCE_TREE/model/vit-b-16/pytorch_model.bin"
mkdir -p model
ln -s "$SOURCE_TREE/model/vit-b-16" model/vit-b-16
```

**每种 shot 的权重必须各用自己的正式 E0**。旧 `dto-kd-assets-20261107` Release 中的 `E0_global_KL_model_best.pth` 是本任务 1-shot 权重；不能将它用于 5-shot。5-shot 正式权重如尚未迁到校园端，必须先单独迁移，下方列出原服务器位置。脚本拒绝其他 SHA，不会悄悄换成当前重训权重。

| shot | 原服务器 checkpoint | SHA256 |
| --- | --- | --- |
| 1 | `output/ablation_frame_tsm/ssv2_small_1shot_E0_gpu0_20261021_095537/E0/model_best.pth` | `78aae7edcbea1f5a48f17aa28d9631ccf680f69c08535b40083cfaa4a46cf5cf` |
| 5 | `output/table_reproduction/ssv2small_5shot_exact_resume_20261106_120801/training/model_best.pth` | `8ca0323a80fd7573fa581d50ecae3a81bd7d3bac2a7d594c4919fdef04c31d76` |

两文件分别为 406375259 / 406367003 bytes。源路径相对于 `/home/zyb/TS-FSAR/`；代码包不包含它们，也没有替你确认校园端已具备 5-shot 文件。不要启动训练以补这个文件。

## 校园端验证与正式运行

先做 CPU 数值测试，不加载编码器：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m unittest discover -s research/ps_ds_subspace_experiment/tests -v
```

数据目录应直接包含 `30109.mp4` 等视频，而不是仅包含解压后的外层目录。根据实际位置填写两种 checkpoint。以下选用的 GPU 必须是**当时实际空闲且你有权使用**的卡；脚本不会停止已有进程，也不会调整 DTO 队列。8 张卡不会自动合并用于一个 Episode。

```bash
# 以实际文件位置替换这三项。
export DATA_ROOT=/data/fsar_workspace/datasets/SSv2-Small
export E0_1SHOT=/data/fsar_workspace/weights/E0_global_KL_model_best.pth
export E0_5SHOT=/data/fsar_workspace/weights/E0_5shot_model_best.pth
test -f "$DATA_ROOT/30109.mp4"
test -f "$E0_1SHOT"
test -f "$E0_5SHOT"

RUN=/data/fsar_workspace/psds_results/seed0_fixed
mkdir -p "$RUN"

# GPU 编号仅为命令示例；先确认卡空闲，再选择对应编号。
nohup bash research/ps_ds_subspace_experiment/run_gpu.sh \
  --shot 1 --gpu 0 --data-root "$DATA_ROOT" --checkpoint "$E0_1SHOT" \
  --output "$RUN/shot1" > "$RUN/shot1.console.log" 2>&1 < /dev/null &
echo $! > "$RUN/shot1.pid"

nohup bash research/ps_ds_subspace_experiment/run_gpu.sh \
  --shot 5 --gpu 1 --data-root "$DATA_ROOT" --checkpoint "$E0_5SHOT" \
  --output "$RUN/shot5" > "$RUN/shot5.console.log" 2>&1 < /dev/null &
echo $! > "$RUN/shot5.pid"

tail -n 30 -F "$RUN/shot1.console.log" "$RUN/shot5.console.log"
```

Ctrl+C 只退出 `tail`，后台任务继续。每 10 个 Episode 打印累计四项 Accuracy 和额外耗时；`smoke/explore/formal` 子目录保留独立预测与完成状态。探索前 200 个 Episode 也是正式 10000 个的前缀，不用其标签/准确率修改方法。

脚本启动前验证官方源码/配置、CLIP 全部文件、checkpoint SHA、Manifest 和依赖文件 SHA。校园端前三个 Episode 的像素 SHA 必须与源端审计一致。每个 Episode 的 E0 结果还和原 `TS_FSAR.forward` 的 loss/accuracy 精确对照。原模型参数/缓冲区以及 checkpoint 前后 SHA 不变才写 `completion.json`。

正式配置固定 5-way、1/5-shot、每类 1 Query、8 帧、24 Novel 类。不要自行通过参数改成 5 Query/class。当前实现全冻结，没有 AMP、训练、模型优化器或新 checkpoint 保存。

若只想分阶段执行，直接调用：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m research.ps_ds_subspace_experiment.evaluate \
  --stage smoke --shot 1 --data-root "$DATA_ROOT" --checkpoint "$E0_1SHOT" \
  --manifest research/ps_ds_subspace_experiment/results/shot1/episodes.jsonl.gz \
  --output "$RUN/shot1"
# smoke 完成后依次换成 --stage explore / --stage formal。
```

`--disable-psds` 一键精确回到原 E0；请用新的 `--output`，不能把该模式混进 PS+DS 运行。运行同一配置和输出目录可续跑已完成的 Episode；完整 JSON 行逐条核对 Manifest 前缀。并发使用同一根目录会被锁拒绝。若最后一行因外部断电被截断，脚本明确报错，请保留原文件后人工检查，不会自动删记录重抽样。

## 统计与结果

两种 shot 的正式任务完成后：

```bash
python -m research.ps_ds_subspace_experiment.analyze \
  --shot1 "$RUN/shot1/formal" --shot5 "$RUN/shot5/formal" \
  --bootstrap 10000 --output "$RUN/formal_analysis"
```

输出准确率表、两种 shot 的统计 JSON、逐 Query 预测变化 CSV 和报告。对探索结果可用 `explore` 替代 `formal`，选择另一个新的分析目录；报告会明确标为探索。

默认 10000 次 Episode-level paired bootstrap，至少 5000 次。Matching/Fusion 分别给出 pp、CI、共同正确/错误、纠正/新增错误、Hard Negative、实际额外耗时、谱秩、正交误差和尺度统计。GPU 单 Episode 延迟以实测为准；合成 CPU 时间不能外推 A10。额外耗时是配对实现中模板拟合+投影匹配开销，尚未减去单独部署 E1 时可省去的 E0 匹配计算。

完整结果文件一般需要几百 MB，两种 shot 建议预留至少 2 GiB 的 `/data` 输出空间，代码包约数 MB，checkpoint/CLIP/视频另计。不会保存视频特征大缓存。

如果校验、解码、数值、秩、正交性、原版 parity 或磁盘检查失败立即停止；保留日志。不要改 Manifest、筛 Episode、放宽 Query 数量或用 Test 准确率重新选择 r/lr/步数。如果方法低于 E0，保留并报告该结果。
