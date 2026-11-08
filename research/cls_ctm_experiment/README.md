# CLS-CTM：保留跨类统计，加入最近竞争类判别信息

分支：`research/cls-ctm-discriminative`。入口独立于原模型、DTO-KD、SAM/QAM
和历史结果；新方案不是官方 CTM/TDM 的完整复现。

## 唯一方案

1. Support 的 LSN CLS 为 `[5*K,8,256]`，先做帧均值，再按 Support 标签做
   shot 均值，得到五个类别原型 `u_c`。不读取 Query，也不读取 Query 标签。
2. 保留成功 CTM 适配版的跨类均值 `mu=mean_c(u_c)` 和总体方差
   `v=mean_c((u_c-mu)^2)`，不做额外归一化。
3. 对每类，以完整 CLS 原型的余弦相似度寻找最相近的其他类别 `n(c)`。
   计算逐通道平方差 `d_c=(u_c-u_n(c))^2`，再平均得到 `d=mean_c(d_c)`。
   余弦完全相同时，平均所有最近候选的平方差，保证类别重排等价。
   仅选择竞争类别时使用归一化原型；平方差使用原始 CLS，保持与原方差同量纲。
4. `[mu;v;d]` 为 768 维，输入 `Linear(768,256) -> GELU -> Linear(256,256)`。
   输出 `m=256*softmax(logits)`，最终 `g=1+0.1*(m-1)`。
5. 最后一层权重和偏置全部零初始化，初始 `g=1`。不加入 BN、训练噪声、
   Query 条件分支、Attention、独立分类分数或新的辅助损失。
6. **每个 Episode 只有一个 `[256]` 权重向量**，共享给所有 Support、Query
   的 CLS/Patch 及所有八帧。加权后执行原 CLS–Patch 加权、原 alpha-D、
   原帧间余弦及 shot 均值原型。TSM 始终从未调制的 CLS 生成。

新模块共有 262,656 参数。旧 CTM 的 512 维摘要完整保留，新增的 256 维
最近竞争类摘要是本次工程设计。它衡量原型间分离，不是可靠性估计，也不称作
类内方差；1-shot 不伪造跨视频类内统计，八帧也不作为八个独立 shot。

为同一 Episode 的共享 gate，每个视频的 alpha-D 只计算一次，不采用旧
SAM/QAM 的逐 Query、逐候选类重复计算。训练使用按帧分块的 non-reentrant
activation checkpointing；这只改变计算内存，不改变数学定义。

## 训练及评测

- 冻结原 E0 全部参数，`eval()`，分别使用已锁定 SHA256 的 1/5-shot E0。
  仅新模块可训练。与历史 CTM 的联合训练不同，不宣称完全复现历史涨点。
- 在官方 64 Base Train 类采样 5-way、对应 shot、每类 5 Query、8 帧，
  使用原训练预处理。模块以原 matching CE 学习；本次没有增加融合训练损失。
- 固定 seed=0、10,000 个训练 Episode，AdamW，LR=0.00025，WD=0.1，cosine，
  无 AMP；初始化后先做两个临时训练更新和三个真实视频 smoke，正式模块重新
  初始化。临时模块不参与正式结果。每 500 步保存精确续训状态。
- 正式训练期间不做周期 Novel-Test；使用固定预算的最终模块，不用 Test 选模。
  完成后依次 smoke 3、探索 200、正式 10,000 Episode；探索结果不用于调参。
- 每个 Episode 同时计算原始完整 E0 与改进版，复用旧 PSDS/SAMQAM 的固定
  官方 24 Novel-Test 类 Manifest、视频和帧、预处理、Query 数量及标签映射。
- **Fusion 为主要指标**，Matching 为辅助指标。CLIP/Side 概率逐 Episode 复用
  原 E0；保持原每 shot 对应的乘积融合公式、指数及温度，不做测试集校准。
- 报告纠正/新增错误、Episode 配对 bootstrap CI、Hard Negative、gate 统计、
  分数尺度和实际新增耗时。两种 shot 完成后自动生成最终报告。

## 校园服务器启动

在新 Terminal 中执行。旧 GPU 任务不会被停止；两张指定卡任一被占用时拒绝启动。
下面使用 GPU4、5；如仍在跑 SAM/QAM，应等待它们完成，或指定两张实际空闲卡。

```bash
(
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd
export PATH="$HOME/.local/gh-cli/bin:$HOME/.local/bin:$PATH"

TREE=/data/fsar_workspace/TS-FSAR-CLSCTM
if [[ ! -d "$TREE" ]]; then
  git clone --single-branch --branch research/cls-ctm-discriminative \
    https://github.com/baizeCrystal/TS-FSAR.git "$TREE"
fi
cd "$TREE"
test "$(git branch --show-current)" = research/cls-ctm-discriminative
git pull --ff-only
bash research/cls_ctm_experiment/launch_campus.sh --gpus 4,5
)
```

启动脚本自身将 **CPU 检查、GitHub 资源校验、GPU smoke、训练和评测整个流程**
放入后台，Terminal 断开不影响流程。日志绝对路径会立即打印。
资源沿用 `/data/fsar_workspace/psds_assets/`，SHA256 通过的文件不会重新下载。
新 clone 自动恢复模型目录和软链接；不复制数据集、不重新打包模型。
GitHub 私有 Release 资源仍需已登录 `gh auth status`，不在脚本保存凭据。

启动早期先查看命令打印的 `<output-root>.launcher.log`；准备完成后可执行：

```bash
RUN="$(cat /data/fsar_workspace/latest_tsfsar_clsctm_root.txt)"
cat "$RUN/status.json"
tail -n 40 -F "$RUN/shot1.console.log" "$RUN/shot5.console.log"
```

只看 1-shot 的训练和测评进度：

```bash
RUN="$(cat /data/fsar_workspace/latest_tsfsar_clsctm_root.txt)"
tail -n 40 -F "$RUN/shot1.console.log" | \
  grep --line-buffered -E '^CLSCTM_(PHASE|TRAIN |PAIRED|EVALUATION_COMPLETE|PIPELINE_COMPLETE)'
```

结果位于 `<output-root>/formal_analysis/`：

- `accuracy_table.md`：两个 shot 的 Matching/Fusion E0/E1 表。
- `shot{1,5}_statistics.json`：配对 CI、错误变化、Hard Negative、实际耗时。
- `shot{1,5}_prediction_changes.csv`：逐样本预测变化。
- `experiment_report.md`：最终报告，Fusion 是主要指标。

失败后保留日志和检查点。精确续训整个某一 shot 的流程：

```bash
TREE="$(cat /data/fsar_workspace/latest_tsfsar_clsctm_tree.txt)"
RUN="$(cat /data/fsar_workspace/latest_tsfsar_clsctm_root.txt)"
SHOT=1
GPU=4
CKPT="$(python - "$SHOT" <<'PY'
import json, sys
print(json.load(open('/data/fsar_workspace/psds_assets/PSDS_READY.json'))['checkpoint_paths'][sys.argv[1]])
PY
)"
cd "$TREE"
nohup bash research/cls_ctm_experiment/run_gpu.sh \
  --shot "$SHOT" --gpu "$GPU" \
  --data-root /data/fsar_workspace/datasets/SSv2-Small \
  --checkpoint "$CKPT" \
  --manifest "$TREE/research/ps_ds_subspace_experiment/results/shot${SHOT}/episodes.jsonl.gz" \
  --output "$RUN/shot${SHOT}" \
  > "$RUN/shot${SHOT}.resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

代码、输入 checkpoint、配置、训练随机状态均参与恢复检查，不允许把新代码
续进旧方法检查点。输出文件锁和校园 GPU 锁阻止重复启动。

## 关闭模块与本地验证

`matching(..., enabled=False)` 直接调用原始 DCM；评测入口支持
`--disable-clsctm`，使用独立输出目录和完整 smoke/explore/formal 阶段。
恒等初始化的 enabled 路径也与原始 DCM 做数值一致性检查。

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover \
  -s research/cls_ctm_experiment/tests -v
```

本地测试使用真实 TS_DCM/alpha-D，合成 LSN 特征，无视频编码器 GPU 推理。
覆盖两种 shot、原始路径一致性、非均匀权重公式、TSM 原始条件、Support/类标
置换、Query/Patch 不参与 gate、最近类并列处理、梯度与 AdamW 更新一致性、
256 通道/196 Patch/8 帧反向，以及检查点/统计/忙 GPU 拒绝。实际 GPU 时延、
显存和准确率需校园执行；这里没有预先生成任何实验成绩。
