# 官方评测协议核对与执行（2026-11-07）

本次以公开官方仓库 https://github.com/yaozhang0417/TS-FSAR 的提交
`1a9105323b3e8cbf51a3ea59768a3d4618bd884f` 为基准，直接下载并逐文件核对。
`assets/OFFICIAL_EVALUATION_SOURCE_SHA256.json` 锁定该提交 81 个源码、配置、划分、prompt 和脚本文件。
此前 v2 不是官方选模/采样协议；基础超参数一致不能被表述为完整官方复现。
用户现已明确要求官方处理方式，覆盖此前 Meta-Val 方案；旧文件仅保留历史身份。

## 实际官方流程

官方 `main.py:main` 在训练中调用 `build_loader(cfg, "test")`，并不是独立 Val。
SSv2-Small 配置对应 24 类 `test_few_shot.txt` 与原 `Ssv2small_test.json`。
训练 10000 个 episode，每 500 个 episode 调用 `test()`，每次 10000 个 episode。
评估在当前 episode backward 后、optimizer step 前执行，按融合准确率保存 best。
单独 Metatest 使用对应 shot 的官方 `Metatest_{shot}shot.yaml`，严格加载指定 checkpoint，
同样评估 10000 个 episode、每类 1 个 query、5-way；8 帧与原预处理不变。
1-shot 融合 Vision/CLIP/Side = 0.4/0.2/0.4，5-shot = 0.8/0.05/0.15。

官方 `test()` 无 autocast，使用 `np.mean(test_accs)*100`。官方 loader 未固定逐 episode
种子、没有固定 episode bank，每次迭代按原 worker 随机流采样。
不能额外固定 10000 个 episode 后仍称为官方逐行流程。
公开实现确实使用 Test 选 best；复现时保留这一规则，并如实标注这一限制。

## 与当前校园运行的差异

| 项目 | 官方公开代码 | 现有校园 v2 |
|---|---|---|
| 周期评估/选 best | 24 类 Test | 12 类 Meta-Val |
| 周期评估文本 | 原 Test LLM prompt | 固定 Val 类别名 prompt |
| 随机数/episode | 官方 loader/worker 随机流 | 每个 episode 独立种子、固定输入 |
| 融合/帧/way/query 数 | 原配置 | 基础配置相同 |

当前 63.040% 是新 v2 的 Meta-Val，不是历史 60.774% 所在的 Test。
历史 60.774% 又不同于固定 Test episode 重评 60.660%。
当前模型可用下述官方入口测试，但训练期选模仍是 Meta-Val。
只重测一个 Val-best checkpoint 不能恢复 Test-best，不能称完整官方训练复现。
要求完整官方复现时，E0 及对照均需新目录从头按官方选模/采样协议运行；
不可中途换 split 后拼接旧曲线，也不可让 E0 与 E1 用不同选模流程。
本次没有连接校园服务器、停止任务或启动新的训练。

## 校园端官方单 checkpoint 测试

新入口直接获取固定提交的独立官方代码副本，核对 81 个文件的 SHA256，
使用官方原样 `main.py`、Dataset、loader、模型、融合与 Metatest 配置。
只替换数据根路径、checkpoint 路径和输出路径，不导入本分支 v2 evaluator。
官方 seed、episode 数、worker、prompt 和融合配置全部保留。
权重先复制为独立评测快照并核对 SHA256，避免训练中更新 model_best 导致读到不同权重。
CLIP 全部文件按既有哈希清单验证，严格加载不兼容的 checkpoint 会由官方入口报错。
本脚本仅支持评测，不启动训练。默认只准备；`--execute` 才执行 GPU 测评。

校园使用新 worktree 获取本次工具，避免覆盖正在运行的源码。将本文件所在提交
检出为新 worktree；模型用 `--clip-root` 指向已有经过验证的 CLIP 文件夹。
环境仍使用 `/data/fsar_workspace/envs/tsfsar-dto-kd`；A10/环境与作者机器不同，不能承诺逐位或精度完全一致。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd

# 在包含本工具的新代码目录执行。GPU 必须是当前已分配且空闲的卡。
OUT="$PWD/runs/campus/official_eval_E0_1shot_$(date +%Y%m%d_%H%M%S)"
python -u scripts/run_official_evaluation.py \
  --shot 1 --gpu 2 \
  --dataset-root /data/fsar_workspace/datasets/SSv2-Small \
  --clip-root /data/fsar_workspace/datasets/TS-FSAR-blocking/TS-FSAR/model/vit-b-16 \
  --checkpoint /实际已锁定的checkpoint/model_best.pth \
  --output "$OUT" --execute
```

这里 GPU 2 仅为命令示例，当前已知有 E1_safe 任务，不能直接占用。
工具会检查共同 GPU 锁与 `nvidia-smi`，发现占用就报错，不停止已有任务。
5-shot 用 `--shot 5` 及其真正的 5-shot checkpoint；E1_safe 权重同样从这个官方推理入口评测。
需要后台运行时用 `nohup ... > driver.log 2>&1 < /dev/null &`，
在另一个终端 `tail -F "$OUT/console.log"`。
官方 tqdm/原始日志原样保留，末尾为 `Final Test Accuracy`；精度在官方输出中四舍五入为两位。
`official_evaluation.json` 记录源码、CLIP、checkpoint、配置哈希与退出状态，不编造精确位数或配对 CI。
每次必须使用新输出目录，保留失败现场；每次约额外占 checkpoint 大小（约 388 MiB）加少量源码/日志。

## 当前验证边界

原服务器仅 CPU/文件核对与单元测试、真实文件的准备流程；未导入 torch、加载模型或执行 GPU 测评。
4 项新增 CPU 测试通过。真实 E0 checkpoint（SHA256 `78aae7edcbea1f5a48f17aa28d9631ccf680f69c08535b40083cfaa4a46cf5cf`）
已在 `runs/official_evaluation_audit_20261107/E0_1shot_prepared/` 完成准备：
81 个官方源码文件、完整 CLIP 哈希、2400 个官方 Test 视频文件及 checkpoint 快照均检查通过。
通过准备不表示校园完整模型兼容性/准确率已验证，需以校园官方入口实际成功退出为准。
本次工具解决独立 checkpoint 的官方测评；它不会修改旧 v2 训练或其历史选模规则。
