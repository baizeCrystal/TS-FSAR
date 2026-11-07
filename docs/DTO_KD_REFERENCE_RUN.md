# DTO-KD 论文参照版本：独立检查与运行

本次修正有两个独立部分：

1. `226dba9` 修复评测器遗漏 import json 导致第1000个 Episode 的 EVAL_PROGRESS 报错，并让所有新短验收主动覆盖该打印分支。
2. 新 E1_amortized/E1_exact 保留论文原始 log-loss，与旧 E1_safe 明确分离。方法细节见 DTO_KD_METHOD_ALIGNMENT.md。

旧 official-test 队列因工程异常中断的记录必须保留，不能当完整训练结果；旧训练代码目录禁止直接 pull 或热修改。
新代码必须置于独立 worktree，旧通过报告仅复用其初始模型/RNG及哈希，全部当前代码 CUDA 验收重做。
当前原版官方 worker 随机流不能精确恢复，不能把已有的 model_best 或不完整断点伪装成同预算续训。

## 快捷入口

`setup_campus_dto_reference.sh --revision <完整提交哈希> --stop-known-queue --gpus 0,2`
会在独立 worktree 中先核对权重、执行 CPU 回归与初始权重哈希预检，随后只停止指针指向的旧正式队列、等待GPU释放，再后台启动新验收。
它不改旧训练源码、不删除输出、不自动启动正式训练。去掉 --stop-known-queue 则不停止任何旧任务，所选卡仍忙会报错。
脚本应从该固定提交通过 git show 提取到 /tmp 后执行，避免在旧树里 git pull。

## 校园：只执行新验收，不自动启动正式训练

先用针对旧 status.json 的 stop_campus_jobs.py 停止该旧队列剩余任务（保留全部文件），再执行下面脚本。
脚本会读取之前记录的最新 official-tree 路径，建议不要手工覆盖该指针。新路径另存 latest_tsfsar_dto_reference_tree.txt。

```bash
(
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd
export TSFSAR_SSV2_ROOT=/data/fsar_workspace/datasets/SSv2-Small
export TSFSAR_RUNTIME_TMPDIR=/tmp TSFSAR_CPU_THREADS=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

OLD="$(cat /data/fsar_workspace/latest_tsfsar_official_tree.txt)"
PREVIOUS="$(cat "$OLD/runs/campus/latest_official_test_root.txt")/acceptance"
git -C "$OLD" fetch origin research/dto-kd-fsar-gradient-coordination
NEW="${OLD}-dto-reference-$(date +%Y%m%d_%H%M%S)"
git -C "$OLD" worktree add --detach "$NEW" FETCH_HEAD
mkdir -p "$NEW/model" "$NEW/runs/campus"
ln -s "$OLD/model/vit-b-16" "$NEW/model/vit-b-16"
cd "$NEW"
python scripts/verify_weights.py
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_dto_kd_reference_cpu tests.test_evaluation_progress_cpu -v

RUN="$NEW/runs/campus/dto_reference_seed0_$(date +%Y%m%d_%H%M%S)"
echo "$NEW" > /data/fsar_workspace/latest_tsfsar_dto_reference_tree.txt
nohup python -u scripts/run_campus_dto_reference.py \
  --previous-acceptance-root "$PREVIOUS" --gpus 0,2 --seed 0 \
  --output-root "$RUN" --apply \
  > "${RUN}.driver.log" 2>&1 < /dev/null &
echo "验收管理进程 PID：$!"
echo "日志：${RUN}.driver.log"
tail -n 40 -F "${RUN}.driver.log"
)
```

每张 A10 单进程，禁止使用有 ECC 错误的 GPU7。所选 GPU 被占用则报错，工具不杀任何现存任务。
默认仅做验收，输出 BOTH_DTO_REFERENCE_ACCEPTANCES_PASSED 也不会自行启动正式训练。
运行目录已存在则拒绝重试覆盖。Ctrl+C 退出 tail 不停止 nohup 管理器。

## 验收通过后，显式选择正式队列

使用同一代码树和同一对新验收证书。以下命令仅作明确的可执行接口示例，**不由验收工具自动执行**：

```bash
TREE="$(cat /data/fsar_workspace/latest_tsfsar_dto_reference_tree.txt)"
cd "$TREE"
ACCEPT_RUN="$(cat runs/campus/latest_dto_reference_root.txt)"
FORMAL="$TREE/runs/campus/dto_reference_formal_$(date +%Y%m%d_%H%M%S)"
nohup python -u scripts/launch_campus_suite.py \
  --protocol v2 --selection-protocol official-test \
  --experiments E0,E0_detach,E1_amortized,E1_exact \
  --gpus 0,1,2,3,4,5,6 --seed 0 \
  --acceptance-1shot "$ACCEPT_RUN/acceptance/shot1/acceptance.json" \
  --acceptance-5shot "$ACCEPT_RUN/acceptance/shot5/acceptance.json" \
  --activation-offload diagnostics --output-root "$FORMAL" \
  > "${FORMAL}.launcher.log" 2>&1 < /dev/null &
```

上述明确示例为 E0/E0_detach/E1_amortized/E1_exact × 两种 shot 共8任务：
E0→E0_detach 隔离 KL 教师路径；E0_detach→两种 DTO 隔离优化方案整体；
E1_amortized→E1_exact 检验论文近似与真实梯度解的差异。新组不要覆盖旧 E1_safe 或改名冒充同一方法。
如需继续此前已确认的 E2 固定系数控制，显式加入 `,E2`，增加4任务；不能静默取消或冒充这些任务已经完成。
此脚本不会自动校准 E3；旧 E3 管理器只识别旧 E1_safe 来源，不能用错来源。
如需匹配新方法的固定尺度控制，应另按相同教师路径、相同目标/CE定义在 Meta-Train 梯度上预先冻结方案。

所有正式组仍训练10000个 Episode、每500步在官方24类Test测10000个Episode选best，保持原帧数、batch、融合、优化器及选模时序。
Test用于复现官方选模须披露；不能根据这些Test结果反复调新损失、融合或超参数。
学校空间至少30GiB启动门槛不等于个人配额保证。闭式版额外计算和主机内存开销需要在验收/正式日志中实际核对。
