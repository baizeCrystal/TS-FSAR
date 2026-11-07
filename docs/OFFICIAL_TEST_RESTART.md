# 所有版本重启：官方 24 类 Test 周期选模

本次用户确认：E0、E1_safe、E2 半倍/两倍 KL，1/5-shot 共 8 个任务；
这些任务全部成功后，自动用新 E1_safe 的早期训练梯度校准 E3，再训练两种 shot。
不重跑已知有适配问题的旧 E1/E1_detach，也不额外启动 E0_detach。

## 这轮冻结的协议

- 训练 10000 个 episode；每 500 个 episode，在官方 24 类 Novel Test 上测试 10000 个 episode并选 best。
- 使用原 `test_few_shot.txt`、原 `Ssv2small_test.json`、原 24 类名称；相关文件与锁定官方提交逐字节校验。
- 5-way、8 帧、train query/class=5、test query/class=1、episode batch=1、梯度累计=1。
- 原训练 RandomSampler、原 DataLoader worker 随机流；每轮测试重新采样，没有逐 episode 独立种子或固定 episode bank。
- 原评估 FP32、原特征/文本归一化、原融合乘积；best 使用官方 `np.mean(test_accs)*100` 的统计方式。
- 保留官方 backward 后、optimizer step 前测试与保存 best 的顺序。
- 1-shot LR=0.00025，5-shot LR=0.0003；原 AdamW、eps=1e-6、weight decay=0.1、原 cosine 时序不变。
- 每种 shot/seed 的全部组从同一份经过哈希核验的**初始**模型重新开始；不加载任何旧训练 best，不恢复旧 optimizer/scaler/controller。
- 原终点模型和新终点模型会有训练随机性/硬件差异；对齐的是公开配方及本轮各组统一流程，不承诺逐位复现作者训练或原60.77。

| 版本 | 1-shot | 5-shot |
|---|---|---|
| E0 | 原 KL=0.1，原 CE 与原教师梯度 | 原 KL=0.01 |
| E1_safe | 已确认的 log1p 适配，KL 教师目标 detach；教师 CE 继续训练 Adapter | 相同规则 |
| E2 低 | KL=0.05；其他同 E0 | KL=0.005 |
| E2 高 | KL=0.2；其他同 E0 | KL=0.02 |
| E3（后续自动） | 新 E1_safe 训练前500步的20个共享LSN梯度点确定固定KL系数 | 独立用5-shot训练梯度校准 |

E3 校准沿用 E1_safe 的 KL 教师目标 detach，教师 CE 保持。它匹配早期 KL/DCM 相对梯度尺度，
不等同于匹配全部监督梯度或完整训练轨迹，不能单独排除所有因果混杂。
系数不设人为截断，不读取 Test 准确率/预测，不从旧 E1 校准；实际值、输入哈希和限制写入报告。
Test 周期选 checkpoint 是按公开实现复现；仍如实披露 Test 选模限制，不能据这些 Test 结果再搜索新损失、融合权重或其他超参数。

## 校园启动

必须创建新 worktree，不能覆盖正在运行的旧训练源码。原服务器不执行训练或操作 GPU。
校园运行以下命令；工具先验证旧 launcher/进程的源码目录、配置路径、owner/session，
只停止列出的两个旧队列，保留全部旧输出。然后等待所选卡退出计算/释放锁，最多等待60秒。
遇到不匹配的运行中任务、非零 ECC、外部任务占卡或空间不足会报错，绝不杀无关进程或使用 SIGKILL。

```bash
(
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd
export TSFSAR_SSV2_ROOT=/data/fsar_workspace/datasets/SSv2-Small
export TSFSAR_RUNTIME_TMPDIR=/tmp TSFSAR_CPU_THREADS=2

OLD=/data/fsar_workspace/datasets/TS-FSAR-blocking/TS-FSAR
SAFE="${OLD}-dto-safe-20261107_145849"
git -C "$OLD" fetch origin research/dto-kd-fsar-gradient-coordination
NEW="${OLD}-official-test-$(date +%Y%m%d_%H%M%S)"
git -C "$OLD" worktree add --detach "$NEW" FETCH_HEAD
mkdir -p "$NEW/model" "$NEW/runs/campus"
ln -s "$OLD/model/vit-b-16" "$NEW/model/vit-b-16"
cd "$NEW"
python scripts/verify_weights.py

RUN="$NEW/runs/campus/official_test_seed0_$(date +%Y%m%d_%H%M%S)"
echo "$NEW" > /data/fsar_workspace/latest_tsfsar_official_tree.txt
nohup python -u scripts/restart_campus_official_test.py \
  --old-run-root "$OLD/runs/campus/rigorous_v2_seed0_20261107_120628" \
  --old-run-root "$SAFE/runs/campus/e1_first_20261107_153656/formal" \
  --previous-acceptance-root "$SAFE/runs/campus/adaptation_20261107_150707/acceptance" \
  --gpus 0,1,2,3,4,5,6 --seed 0 --output-root "$RUN" --apply \
  > "${RUN}.driver.log" 2>&1 < /dev/null &
echo "重启管理进程 PID：$!"
echo "新代码：$NEW"
echo "总日志：${RUN}.driver.log"
tail -n 40 -F "${RUN}.driver.log"
)
```

GPU7 曾出现不可纠正 ECC，继续排除；7 张健康卡最多7个任务同时运行，第8个自动排队。
每进程单卡，保持原 batch；不能把8卡显存相加来扩大单进程batch。
CPU activation offload 仅诊断时启用，沿用已验证的 A10 显存修复，需在这轮校园验收中复核。
这里读取旧 acceptance 的初始模型/RNG以避免重复PCA；新代码下所有 CUDA 等价、AMP和短训练检查全部重做。
新报告绑定 `official-test`，旧 Meta-Val 报告不能直接授权新正式任务。
验收包含32-episode短运行及2个测试 episode，不是正式10000步结果；失败不会启动正式队列。
`Ctrl+C` 只退出 tail；后台管理进程保持运行。不要重复执行整段创建第二个重启任务。

## 重开终端后看进度

```bash
TREE="$(cat /data/fsar_workspace/latest_tsfsar_official_tree.txt)"
RUN="$(cat "$TREE/runs/campus/latest_official_test_root.txt")"
tail -n 40 -F "${RUN}.driver.log"
```

若 `latest_official_test_root.txt` 尚未出现，先看上面首次启动输出的总日志：仍可能在停止旧任务/等待GPU阶段。
正式训练开始后，1-shot E1 日志：

```bash
tail -n 30 -F "$RUN/formal/E1_safe_1shot/console.log"
```

E0 对应 `$RUN/formal/E0_1shot/console.log`。测试每1000个 episode打印 `EVAL_PROGRESS`，
完成后打印 `TEST_SELECTION`，其 `split=test`、`episodes=10000`、选模来源为 `Novel-Test`。
`validation.jsonl` 文件名为历史兼容名称，其每行真实 split 和选择来源均明确为 Test，不再输出含混的 VAL。
每轮预测/episode ledger 独立保存；Test 随机重采样不能强行跨轮/跨组配对 Bootstrap。

8个任务全部成功后，管理进程自动启动E3，不需要再敲命令。
E3记录位于 `$RUN/E3/`，含两种shot的配置、校准报告、status与训练日志。
若前8个任务失败或中断，E3不启动。若E3校准/训练失败，保留现场，不自动重试或换系数。

## 存储、恢复与验证边界

启动前要求所在文件系统至少30GiB可用，供新初始化、短验收、checkpoint和预测；
这只是保守启动门槛，不代表个人配额或完整存储保证。原始视频和旧实验均保留。
官方多worker随机流不保存完整worker状态，因此本模式拒绝 `--resume`，不伪装精确续训。
`resume_latest.pth`仍保存审计状态，但不能在本模式恢复；失败后保留记录并新目录从头开始。

原服务器仅进行CPU合成测试；真实视频与完整模型、CUDA显存/AMP稳定性必须在校园验收。
CPU验证包含与原 `datasets.builder.build_loader` 的真实采样器/帧增强输出逐张量相等、
主进程RNG相等（0/2worker）、24类Test配置与官方文件一致、每轮重新采样、
随机采样训练截断、原选模时序、best保存、E3训练期校准来源及安全停止身份检查。
该测试不能预言训练精度或保证方法提升。
