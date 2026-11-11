# 非均匀时间采样诊断（冻结 E0，无训练）

代码与原训练/评估入口隔离，GPU 只会在显式执行 `run evaluate` 时启用。
默认两种 shot 都使用同一个、训练于 1-shot 的 E0 checkpoint（SHA256
`78aae7edcbea1f5a48f17aa28d9631ccf680f69c08535b40083cfaa4a46cf5cf`）。
5-shot 结果不能称为独立 70.70% checkpoint 的复现。

先看 [audit.md](audit.md) 的重要边界：现有 E0 是**均匀分段内随机采样**，
不是严格等间隔；本实验优先逐索引复现它，不能既换成 linspace 又声称与历史逐样本一致。
`data/1shot`、`data/5shot` 已打包既有 Test 各前 200 episodes，原顺序不变，
包括原始帧索引、配置、E0 分数/文本概率/输入像素哈希，git pull 即可获取。
历史分析曾检查过这些 Test 数据，结果仅用于探索，不是新的盲测。

## 条件与协议

| 名称 | 定义 |
|---|---|
| `e0` | 精确返回历史 8 个索引（gamma=1） |
| `p075`, `p133` | E1：对记录时间坐标使用 gamma=0.75、1.33 |
| `p05`, `p2` | E2：gamma=0.5、2.0 |
| `r075`, `r133` | E3：轻度幂变换偏移幅度的有序随机对照 |

`--scopes query support both` 分别只扰动 query、扰动**全部** support、两端均扰动，
不根据 Query 标签挑选 Support。所有条件用相同视频、类别、顺序、8 帧和原始融合。

官方 rate=50 的有效范围是 `[0,8*floor(N/8)-1]`（末尾余数帧原来就未被使用）。
令 `u=index/hi`，进行 `u**gamma` 后四舍六入五成双 (`np.rint`) 得到索引。
E0 不重新量化，直接精确保留原索引。短视频 `N<8` 使用 `[0,N-1]`。
不修正/删除重复帧，保存有效帧数和重复比例；没有偷偷改变序列长度。

E3 固定种子，每个视频序列生成 32 组随机符号/置换偏移，排序、截断、量化后，
先选择有效帧数最接近对应 E1、再选择 RMS 时间偏移最接近的候选。
此选择只用采样索引，不接触标签或分数。保存目标/实际 RMS、残差、有效帧数。
**它仍不是严格匹配控制**：时间覆盖、偏移方向、量化和间距分布不完全相同。

缓存单位为整个 8 帧序列，含 CLIP CLS、LSN CLS、真实重新编码后的高维 alpha-D。
键包括视频/split/实际索引/方案/gamma/seed、checkpoint、配置、源码/CLIP 资产哈希、
设备库版本及精度。LSN 含时间注意力，禁止拼接不同序列的单帧缓存来冒充重新采样。
缺失序列按 `--batch-videos` 成批编码；磁盘缓存默认最多 8 GiB，达到预算只计算并使用
256 MiB 内存 LRU，不删除任何旧文件。为不同 run 使用单独输出目录，缓存可以复用。

## 原服务器 CPU 检查

以下命令实际存在并已测试，不启动 CUDA。基础测试只需要 numpy，无需 pytest：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest experiments.temporal_sampling.test_cpu -v
# 有项目模型依赖时，额外检查 synthetic features 下原生推理公式的一致性：
CUDA_VISIBLE_DEVICES='' TEST_MODEL_PARITY=1 python -m unittest experiments.temporal_sampling.test_cpu -v
python -m experiments.temporal_sampling.run sample \
  --protocol experiments/temporal_sampling/data/1shot --limit 1
```

本机已有的兼容 Python 为 `/home/zyb/miniconda3/envs/msqnet/bin/python`。
如本机使用 `.runtime-deps` 的既有 transformers，可在命令前加
`PYTHONPATH="$PWD/.runtime-deps${PYTHONPATH:+:$PYTHONPATH}"`；校园已安装兼容依赖时无需添加。
不要求安装新依赖。

## 校园服务器可复制命令

先激活现有能运行 TS-FSAR 的 conda 环境。在仓库根目录执行：

```bash
git pull --ff-only origin research/blocking-validation-20261106
export VIDEO_ROOT=/home/zyb/datasets/TS-FSAR/smsm_small
export E0_CHECKPOINT="$PWD/output/ablation_frame_tsm/ssv2_small_1shot_E0_gpu0_20261021_095537/E0/model_best.pth"
export CUDA_VISIBLE_DEVICES=0
export PROTOCOL=experiments/temporal_sampling/data/1shot
# 校园的数据/checkpoint 如在不同位置，只调整上面两个路径。
# 每次设置一个新的根目录；其下每个评估目录必须不存在。
export TASK_RUN="$PWD/output/temporal_sampling_$(date +%Y%m%d_%H%M%S)"

# 1. 只读检查资产、所有视频路径、checkpoint SHA256、Episode 一致性；不运行模型。
python -m experiments.temporal_sampling.run preflight \
  --protocol "$PROTOCOL" --checkpoint "$E0_CHECKPOINT" --video-root "$VIDEO_ROOT"

# 2. 单视频：所有采样方案与实际解码形状（CPU）。
CUDA_VISIBLE_DEVICES='' python -m experiments.temporal_sampling.run sample \
  --protocol "$PROTOCOL" --limit 1 --decode --video-root "$VIDEO_ROOT"

# 3. 单 Episode GPU 冒烟。先 E0，再轻度 query-only。
python -m experiments.temporal_sampling.run evaluate \
  --protocol "$PROTOCOL" --checkpoint "$E0_CHECKPOINT" --video-root "$VIDEO_ROOT" \
  --limit 1 --schemes e0 p075 --scopes query --max-minutes 10 \
  --output "$TASK_RUN/smoke" --cache "$TASK_RUN/cache"

# 4. 200 Episode 的 E0：全部像素、分数、预测与历史核对。
python -m experiments.temporal_sampling.run evaluate \
  --protocol "$PROTOCOL" --checkpoint "$E0_CHECKPOINT" --video-root "$VIDEO_ROOT" \
  --limit 200 --schemes e0 --output "$TASK_RUN/baseline" --cache "$TASK_RUN/cache"

# 5. E1 Query-only 和 Support-only（先重新核对全部 E0，再执行干预）。
python -m experiments.temporal_sampling.run evaluate \
  --protocol "$PROTOCOL" --checkpoint "$E0_CHECKPOINT" --video-root "$VIDEO_ROOT" \
  --limit 200 --schemes e0 p075 p133 --scopes query support \
  --output "$TASK_RUN/e1" --cache "$TASK_RUN/cache" --batch-videos 2

# 6. CPU 离线分析和代表案例。后者提供视频路径时会额外渲染真实帧。
CUDA_VISIBLE_DEVICES='' python -m experiments.temporal_sampling.analyze \
  --run "$TASK_RUN/e1" --output "$TASK_RUN/e1_analysis"
CUDA_VISIBLE_DEVICES='' python -m experiments.temporal_sampling.visualize \
  --run "$TASK_RUN/e1" --cases "$TASK_RUN/e1_analysis/cases.json" \
  --video-root "$VIDEO_ROOT" --output "$TASK_RUN/e1_figures" --max-cases 5
```

换成 `export PROTOCOL=experiments/temporal_sampling/data/5shot`，使用新的 `TASK_RUN`，
再运行同样步骤即可做 5-shot。相同 checkpoint 和原融合 .4/.2/.4 均不变。

**可选、不会自动执行** E2/E3/Both（先确认 E0/E1 正常）：

```bash
python -m experiments.temporal_sampling.run evaluate \
  --protocol "$PROTOCOL" --checkpoint "$E0_CHECKPOINT" --video-root "$VIDEO_ROOT" \
  --limit 200 --schemes e0 p075 p133 p05 p2 r075 r133 --scopes query support both \
  --output "$TASK_RUN/extended" --cache "$TASK_RUN/cache" --max-minutes 55
CUDA_VISIBLE_DEVICES='' python -m experiments.temporal_sampling.analyze \
  --run "$TASK_RUN/extended" --output "$TASK_RUN/extended_analysis"
```

每次评估默认 55 分钟硬计时（包含模型初始化和 E0），最多允许 60 分钟。
超时写 `partial_timeout` 并返回 124；失败写 `failed`；已经完成的单 Episode 输出保留。
分析默认拒绝不完整运行；显式 `--allow-partial` 仅分析同条件完整配对的子集并标注。
不存在自动排队、自动拓展到 1000 episodes 或自动启动多个 GPU 任务的行为。

若以后决定扩大到 1000，原服务器可用 `export_protocol` 导出**同一历史清单的完整前缀**，
将新协议目录传入 `--protocol ... --limit 1000`。本次只打包 200，不生成新随机 Episode：

```bash
python -m experiments.temporal_sampling.export_protocol \
  --source output/multi_metric_diagnostics/E0_patchcos_otam_20261104_233227 \
  --historical output/tsm_class_discriminability/E0_best_gpu2_20261022_162128 \
  --output output/temporal_sampling_protocol_1000 --limit 1000
```

## 输出与解释

`manifest.json` 保存配置、实际 Git commit、相关源码哈希、checkpoint、协议身份。
`episodes.jsonl` 保存原始视频/标签/索引；`e0_verification.json` 只有全部 E0 核查通过后生成。
`e0_ID.npz`、`SCHEME_SCOPE_ID.npz` 保存四分支真实分数、三分支概率、R/W、CLIP/LSN 帧特征；
配套 JSON 保存采样位置、重复帧、偏移、表示比较。`status.json` 明确成功/失败/超时。

离线生成 `summary.json`、`query_predictions.csv`、`cases.json`、`report.md`。
每个分支独立报告 accuracy、配对 delta、correct→wrong/wrong→correct、预测变化率、margin，
以 Episode 为簇 bootstrap（默认 5000 次，固定 seed）。重复视频和共享 Support 仍可能
使跨 Episode 存在剩余依赖，不能把该 CI 视为消除了所有数据依赖。

特征比较同时保存逐索引余弦距离、公共时间区间内线性插值距离、同源帧重叠位置的距离。
插值只用于诊断，不送回分类器；重复时刻先均值合并，不外推。CLIP 的同源帧应基本稳定；
LSN 同源帧若变化提示序列上下文作用，但不自动表示语义破坏。
不同位置的比较统一称“采样序列级表示变化”，不称语义变化。

R/W 的 margin 分解采用原始固定 rival，精确拆成 `deltaR*W0 + R0*deltaW + deltaR*deltaW`。
这只是原帧索引坐标下的代数分解，不是时间语义对齐，也不是 H1b 的因果证明。
案例按每种转移类别固定顺序选第一个，同时包含保持正确、纠正、破坏和保持错误。
5-shot Support 的 R 轴是同类按位置平均的原型，并非某个真实 Support 视频轨迹。

研究判断模板与未验证项目见 [report.md](report.md)。目前没有本任务的 GPU 结果，
不能回答 H1 成立，也不能据此推荐训练速度归一化模块。
