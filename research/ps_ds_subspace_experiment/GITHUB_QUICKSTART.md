# 从 GitHub 直接恢复 PS+DS 评测

不需要新建仓库。代码位于现有公开仓库的独立分支：

<https://github.com/baizeCrystal/TS-FSAR/tree/research/ps-ds-subspace>

固定 Episode 清单及历史 5-shot 权重位于现有私有数据中转库：

<https://github.com/baizeCrystal/TS-FSAR-Dataset-Transfer/releases/tag/psds-assets-20261108>

1-shot E0、CLIP 复用 `dto-kd-assets-20261107`，不重复上传。模型权重、数据集视频和完整 Episode 文件没有提交到公开 Git 历史。本次没有新建仓库，没有改动校园训练目录或队列。私有附件需要能访问该仓库的 GitHub 身份。

## 校园 Terminal：克隆与补齐资产

以下只准备代码和文件，不会启动模型推理或训练。使用新目录，与当前训练 worktree 分开。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd
mkdir -p /data/fsar_workspace/psds_eval
cd /data/fsar_workspace/psds_eval

git clone --single-branch --branch research/ps-ds-subspace \
  https://github.com/baizeCrystal/TS-FSAR.git TS-FSAR-PSDS
cd TS-FSAR-PSDS

gh auth status
python -u -m research.ps_ds_subspace_experiment.github_assets \
  --assets-dir /data/fsar_workspace/psds_assets
```

如果 `gh auth status` 提示未登录，先执行 `gh auth login --hostname github.com --git-protocol https --web` 完成浏览器授权，再重试。不要在聊天中发送 Token。

准备脚本默认下载：两种 shot 的固定 10000 Episode 清单、各自正式 E0 checkpoint、CLIP。每个文件执行 SHA256 检查；固定清单只解压到当前新 clone 的 `research/ps_ds_subspace_experiment/results/shot{1,5}/`，CLIP 只在新 clone 的 `model/vit-b-16` 创建链接。不下载数据集视频、不安装/更改环境、不运行编码器。

若已经有对应文件，可用可选参数复用，避免重复下载：

```bash
python -u -m research.ps_ds_subspace_experiment.github_assets \
  --assets-dir /data/fsar_workspace/psds_assets \
  --clip-source /实际位置/model/vit-b-16 \
  --e0-1shot /实际位置/E0_global_KL_model_best.pth \
  --e0-5shot /实际位置/E0_5shot_model_best.pth
```

仅保留你实际具备的参数。指定文件必须与锁定 SHA 相同，否则报错。1-shot/5-shot 不混用；本次新上传的 5-shot SHA 为 `8ca0323a80fd7573fa581d50ecae3a81bd7d3bac2a7d594c4919fdef04c31d76`。

重复执行会复用校验通过的文件。`gh release download` 不保证字节级续传；失败下载保留在资产目录下 `.gh-download-*`，未通过校验的文件不会被当作完成，也不会自动删除。默认全部下载并保留压缩包约需 1.8 GiB 磁盘，建议给 `/data` 预留至少 3 GiB，真实数据集和评测输出另计。

准备完成后生成 `/data/fsar_workspace/psds_assets/PSDS_READY.json`，记录通过 SHA 校验的实际权重与 CLIP 路径。

## CPU 检查与 GPU 评测

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m unittest discover -s research/ps_ds_subspace_experiment/tests -v

export E0_1SHOT="$(python -c 'import json; print(json.load(open("/data/fsar_workspace/psds_assets/PSDS_READY.json"))["checkpoint_paths"]["1"])')"
export E0_5SHOT="$(python -c 'import json; print(json.load(open("/data/fsar_workspace/psds_assets/PSDS_READY.json"))["checkpoint_paths"]["5"])')"
export DATA_ROOT=/data/fsar_workspace/datasets/SSv2-Small
test -f "$DATA_ROOT/30109.mp4"
```

数据根目录必须直接包含视频 `.mp4`。接下来按照 [README.md](README.md#校园端验证与正式运行) 中的 `run_gpu.sh` 命令，选**当时空闲**的卡分别运行两种 shot。GPU 编号不由准备脚本自动占用，不会停止现有任务。原有数据集无需重新下载。

若要先看少量 GPU 验收：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m research.ps_ds_subspace_experiment.evaluate \
  --stage smoke --shot 1 --data-root "$DATA_ROOT" --checkpoint "$E0_1SHOT" \
  --manifest research/ps_ds_subspace_experiment/results/shot1/episodes.jsonl.gz \
  --output /data/fsar_workspace/psds_results/shot1
```

GPU 0 仅为示例，请先确认空闲。smoke 通过后，同一输出目录依次 `--stage explore`、`--stage formal`。两种 shot 的最终统计使用 `analyze.py --bootstrap 10000`。当前尚无 PS+DS 的真实 GPU Accuracy，不能将历史 60.77/70.70 写作本次配对结果。
