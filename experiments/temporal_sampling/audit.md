# 实施前只读审计

检查分支 `research/blocking-validation-20261106`，起始 HEAD `688685c`。
现有 worktree 的 `configs/base.yaml`、`datasets/base_dataset.py`、
`datasets/video_few_shot.py`、`main.py`、`trainers/ts_fsar.py` 有用户未提交修改。
本任务不改动/提交这些文件，也不依赖未跟踪的其他实验目录。

1. **解码/采样**：`BaseVideoDataset._decode_video` 使用 decord `VideoReader.get_batch`；
   `_get_video_frames_list` 选择 interval/segment；SSv2 当前有效配置
   `SAMPLING_MODE=interval_based, SAMPLING_RATE=50, NUM_INPUT_FRAMES=8`。
   `_interval_based_sampling` 对每个 `N//8` 长度的分段调用 `random.randint`，
   因而即使测试模式也不是严格等间隔；尾部 `N%8` 帧不在候选范围。
   `_segment_based_sampling(random_sample=False)` 才是分段中点，但本配置未调用它。
2. **预处理**：`Video_few_shot.get_seq` → `_config_transform` 的测试 Compose：
   uint8 `ToTensorVideo` → `KineticsResizedCropFewshot` → Normalize → TCHW。
   `datasets/augmentations.py::_get_controlled_crop` 实际把 H/W 分别 resize 为256，
   然后居中裁剪224；不能换成保持长宽比 resize。
3. **视觉缓存**：官方 get_seq/模型主路径没有视频路径键的预提取特征缓存。
   已有 output 诊断缓存确实存在，但缺少干预后的重新编码输入，不能用于新条件。
4. **LSN/CLIP**：`trainers/model.py::LSN.forward` 对 CLIP 各层特征下投影并与侧网络融合。
   `ViTBlock.forward` 的 temporal 层（本配置6/8/10）在每个视频的8帧间注意力；
   返回 LSN 全部token与投影 CLIP CLS。原始 CLIP帧特征按512维CLS记录，
   LSN按256维CLS记录，alpha-D保留 signed CLS–Patch 加权后的完整真实输出。
5. **alpha-D/匹配**：`trainers/ts_fsar.py::Alpha_DC` 对256通道距离取0.4次幂、
   `1e-5`、双中心化、含对角上三角（32896维）。`TS_DCM.forward` 先对同类
   Support 的 alpha-D 逐位置求均值，再经 `utils.functions.cos_sim(epsilon=.01)`
   构造 `[query,class,query_time,support_time]` 的R。
   `generator(mean(query_CLS)+global_mean(support_CLS))` 产生有符号、query相关、
   候选类共享的8×8 W；匹配分数为64个 `R*W` 的总和。
6. **融合**：`TS_FSAR.forward` eval 将 TS-DCM softmax、Adapter(投影CLIP CLS序列)
   的均值与文本相似概率、LSN CLS 经 up_ln/up_linear 后均值的文本概率，
   按 `Pvision^.4 * Pclip^.2 * Pside^.4` 融合。两个已有固定协议都是相同
   1-shot训练的E0（5-shot仅推理协议变化），不能换用独立5-shot配置的 .8/.05/.15。
7. **Episode**：`Video_few_shot.__getitem__` 原生每次随机采样类/视频；历史工具
   已保存视频、类、Support顺序、Query顺序、帧索引。本任务直接复用
   `output/multi_metric_diagnostics/E0_patchcos_otam_20261104_233227/novel_{1,5}shot`
   的前200，来源均为1000固定Test。原脚本的旧 E1/E2 是指标替换，与本任务时间
   采样 E1/E2 含义不同，**只导入旧E0作为核对参考**。
8. **checkpoint**：
   `output/ablation_frame_tsm/ssv2_small_1shot_E0_gpu0_20261021_095537/E0/model_best.pth`，
   SHA256 `78aae7edcbea1f5a48f17aa28d9631ccf680f69c08535b40083cfaa4a46cf5cf`。
   `load_state_dict(strict=True)`，eval、全部参数冻结、FP32、TF32关闭、无optimizer。

## 原始实现复用与可移植性

当前未提交的 `trainers/ts_fsar.py` 引入多个尚未入Git的实验模块，直接导入会使校园
干净 checkout 失败。`frozen_ts_fsar.py` 是 **`git show 688685c:trainers/ts_fsar.py`
的完整原始实现副本**（仅文件末尾补换行），不是自行重写的模型；其 LSN、cos_sim
仍导入仓库正式已提交实现。副本不会读取当前可选新模块，也不改变原文件。
缓存后只重算原型/文本/匹配读出，首个Episode核对原生 `TS_FSAR.forward` 的
TS-DCM分数、融合acc及loss；每个E0还核对历史全部分支概率/匹配分数/最终预测/像素。
小批次重编码可能产生正常FP32 GEMM差异；逐条件使用缓存前核对E0读出和预测，
任何预测翻转都停止，而不是把它作为采样效应。

## 审计边界

本轮不把真实执行速度和采样分布等同，不研究可变长度片段聚合；
不把表示变化、分类下降、匹配机制失效混为一谈；不使用Test标签选择gamma。
此时 H1/H1a/H1b 均待校园前向验证。
