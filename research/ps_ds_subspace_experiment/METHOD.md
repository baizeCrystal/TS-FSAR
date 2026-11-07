# PS→DS 通道子空间迁移：固定实现

参考：Hu, Cheng, Hauptmann, *Subspace Representation Learning for Few-shot Image Classification*, arXiv:2105.00379v2，2021-05-05，§3.1–3.4、§4.1。
PDF：<https://arxiv.org/pdf/2105.00379>；SHA256 `885574dcd8fdb1f83c2368c16e9916e642eed1b1263bb8fb2091c7ba4cb765f1`。原始目录保留了论文文本提取；公开代码分支只保留论文链接与摘要式方法说明。本实现依据论文公式，不声称已逐行复现作者代码。

## 唯一方案与固定参数

- 通道秩 r=5（论文表 1/2），PS=50 步、lr=0.1，DS=50 步、lr=0.01（论文 §4.1）。两种 shot 完全相同，不根据 Novel Test 结果选择参数。
- 在原 TS_DCM 的 **CLS–Patch cosine 加权之后、α-D 之前**取 `V[BT,256,196]`。对于每个 Support 视频，拼合全部 8 帧的 Patch，形成 `H[256,1568]`。不中心化、白化或另做特征归一化。
- 由 H 的前 r 个**左**奇异向量表示一个视频。使用 float64 的 `eigh(HHᵀ)` 得到同一子空间及奇异值，避免构造大 Vᵀ；数值测试与直接 SVD 对照。论文文字称 right singular vectors，但其 `H[d,hw]` 和 `U[d,s]` 的尺寸要求左奇异向量。
- 每类先拼合该类所有 Support H，进行同样的 SVD，作为该类模板初值；这与论文的 union-of-K-local-feature-sets 初始化一致。
- 模板奇异值权重固定为该类 union SVD 的前 r 个奇异值归一化结果。论文没有明确给出模板 U 经 Cayley 旋转后更新奇异值的规则；此处明确固定权重，仅优化 U。没有额外学习的谱权重。
- **先 PS 再 DS 是本次迁移设计**。论文分别讨论 PS 与 DS，不能将本次串联称为论文原有流程。1-shot 的 PS 仍执行全部 50 步，通常处于自身子空间的驻点。

## WSD 与模板优化

按论文式 (2)，对基 A、B 和各自归一化奇异值 λ、μ：

`D(A,B) = sqrt(1 - Σij sqrt(λi μj) (aiᵀ bj)²)`。

使用奇异值而非平方奇异值。距离平方下限为 `1e-12`，仅避免零距离导数无穷；没有改变温度或搜索这个常数。

PS 最小化 `Σc Σj D(Uc, Hcj)`，保持每个模板列正交。DS 以 PS 的输出初始化，最小化全部 5×shot 个 **Support** 的 `SUM cross_entropy(-D(Hcj, U·), c)`，温度固定 1。保留 sum 而不是 mean，以对应论文求和目标与学习率。

论文式 (5) 将 log posterior 写成正号却称其为要最小化的分类损失；实现使用标准**负**对数似然。式 (6) 中 Ŵ 的尺寸也要求补上 U 的转置。对 Z=∂L/∂U，采用其反对称生成元：

`W = ZUᵀ - UZᵀ - 0.5 U(UᵀZ-ZᵀU)Uᵀ`，

并以**下降**方向更新 `U_next = (I + lr W/2)^(-1) (I - lr W/2) U`。若将论文印刷的正号方向直接用于正梯度，会成为上升方向。用 Woodbury 恒等式把 256 维求解等价化为 2r=10 维求解；与完整 Cayley 解对照测试。没有 QR 替换、跳过关键步骤或提前停止。

WSD 在相同子空间处有尖点，1-shot 尤其明显。固定学习率的 50 步不保证目标单调降低。记录 PS/DS 初末损失、梯度范数、目标上升标志、每步最大正交误差；不选择“最优步数”的模板、不据此改 Test 超参数。合成数据的 1-shot DS 末值略升已如实记录，不把完成优化步骤等同于证明优化有效。

`fit_templates` 只接收 Support 特征和 Support 标签。无 Query 特征、Query 标签或跨 Episode 状态。原模型在 eval 模式，全部参数 `requires_grad=False`；仅新建的 U 使用 `autograd.grad`。

## TS-DCM 类别条件投影

对每个候选类别 c，用 DS 最终 `Uc[256,5]`：

1. Query 的全部帧和该类 Support 的全部帧，分别计算 `Uc(UcᵀV)`，仍为 `[B,256,196]`。
2. 调用原 `Alpha_DC`（α=0.4、原 epsilon、原上三角向量），按原流程对同类 Support 的 α-D 表示取均值。
3. 调用原 `cos_sim`，保留其分母 `||x||||y||+0.01`，形成该类的 8×8 帧间相关性。
4. TSM 的任务原型与 `generator` 全部使用**未投影的原始 CLS 特征**，与 E0 相同；用其该类权重聚合相关性。

逐类填入唯一的 `score_psds[5,5]`。不同候选类别只使用自己的 U。不输出 WSD 分类分数，不融合 PS/DS/WSD 预测，不增加训练损失。

## 完整模型与尺度

CLIP、LSN side 分支与 E0 共用同一次 frozen encoder forward 的特征和分支概率。原概率乘积融合保持：

`p = softmax(matching)^a * p_clip^b * p_side^c`。

1-shot `(a,b,c)=(0.4,0.2,0.4)`；5-shot `(0.8,0.05,0.15)`。没有替换为概率和、调整 temperature、归一化 logits 或用 Query 标签校准。

相关性仍使用原 cosine/TSM，数值接口兼容，但这**不保证**投影前后 logit 分布相同。因此保存两版 logits 的 min/max/std 和 softmax entropy，观察尺度变化及 matching/fusion 表现；不因观察结果改变融合权重。若 matching 提升但 fusion 下降，也应如实报告。

## 官方协议、固定性及统计范围

使用当前正式 `configs/ssv2_small/Metatest_{1,5}shot.yaml` 与原 Config base merge。原版 64/12/24 划分，Test 2400 个视频、24 Novel 类，5-way、8 帧、每类 1 Query、每 Episode 共 5 Query。原官方 interval sampler 在 `SAMPLING_RATE=50` 时，即使 Test 也随机抽分段内帧；本实现冻结其**实际索引**，没有改成均匀中心帧。

Manifest 调用原 `Video_few_shot.__getitem__`，保留原 class/sample 抽样、Support/Query shuffle、标签映射和 frame sampler。快速模式只用已读取的视频长度/FPS替代重复视频解码，Test 中心 resize/crop/normalize 不消耗 RNG；已对每种 shot 的前 3 个 Episode 比对完整原采样器、全部视频/帧/标签，以及解码预处理后的精确像素 SHA。Manifest 的 DataLoader generator 显式固定 seed=0、workers=4，避免建模随机数影响采样流。这是**官方采样规则下的新固定流**，并非历史 best 选模时的原随机状态，不能要求 E0 恰好再得 60.774/70.696。

解码使用 Decord CPU、单线程。原主机自动线程 (0) 出现 FFmpeg EAGAIN；单线程是执行层面的稳定性设置，不更改帧/预处理。解码失败直接终止，禁止重采样 Episode、用相邻视频替换或跳过。

回放只读同一个 Manifest；每个 Episode 只解码一次、encoder 只运行一次，E0/E1 共享同一个图像 Tensor 和 frozen 输出。因此两版 Query、Support、帧和预处理逐字节相同。校园端前 3 个 Episode 还须与源端像素 SHA 一致。每个 Episode 保存像素 hash；两版本预测原子地写入同一配对记录。

smoke=前 3、explore=前 200、formal=全部 10000。正式集合包含探索前缀；没有按探索准确率选择方案/参数/episodes。探索结果不能当最终成绩，也不是独立确认集。

按 Episode 聚合 5 Query 的准确率差，再做默认 10000 次（最少 5000）配对 percentile bootstrap，给出提升 pp 和 95% CI。不将 50000 Query 当独立样本。CI 只覆盖固定 checkpoint 下 Episode 抽样不确定性；不覆盖训练随机种子、历史 Test 选模或方法提出过程。

Hard Negative 预定义为 E0 分数最高的错误候选。报告 E0 错误中 E1 最终纠正数、超过原错误候选但仍误判的情况、错误候选转换及全局类别混淆对；不选择困难度阈值来改善结果。
