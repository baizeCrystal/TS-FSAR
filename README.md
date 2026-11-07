# TS-FSAR: Task-Specific Distance Correlation Matching for Few-Shot Action Recognition

本轮重新训练按 [官方 24 类 Test 重启手册](docs/OFFICIAL_TEST_RESTART.md)：E0/E1_safe/E2 共8组从初始模型重训，官方随机采样与Test周期选best，随后自动校准并补跑E3。此前Meta-Val队列仅保留历史记录。

评测协议更正（2026-11-07）：已有校园 v2 使用独立 Meta-Val 与固定 Episode，**不是官方选模/采样流程**。用户要求以官方为准；官方流程核对及直接运行固定版本官方代码的测试入口见 [官方评测说明](docs/OFFICIAL_EVALUATION_ALIGNMENT.md)。旧 v2 说明仅描述历史运行，不作为完整官方复现承诺。

校园 DTO 实验正式重启使用 [protocol v2 手册](docs/RIGOROUS_RESTART.md)：先停止旧队列、核对基础配方，再执行校园完整模型 CUDA 验收；通过后才运行同初始化/同 Episode 的 E0/E1/E2。旧 softmax-logit 试验与新直接权重更新版本分别保留，不混用结果。

<p align="center">
  <img src="./assets/framework.svg" width="95%"/>
</p>

<p>
  <a href="https://arxiv.org/pdf/2512.11340">
    <img src="https://img.shields.io/badge/PAPER-PDF-A52A2A?style=flat-square&logo=adobeacrobatreader&logoColor=white&labelColor=555555" alt="Paper PDF">
  </a>
  <a href="./assets/poster.pdf">
    <img src="https://img.shields.io/badge/POSTER-PDF-2E8B57?style=flat-square&logo=adobeacrobatreader&logoColor=white&labelColor=555555" alt="Poster">
  </a>
  <a href="https://github.com/yaozhang0417/TS-FSAR">
    <img src="https://img.shields.io/badge/PROJECT-Page-1E90FF?style=flat-square&logo=github&logoColor=white&labelColor=555555" alt="Project Page">
  </a>
</p>


> **Task-Specific Distance Correlation Matching for Few-Shot Action Recognition**[[Paper]](https://arxiv.org/pdf/2512.11340)).
>
> <p>
> <em>
> Fei Long<sup>*1</sup>,
> Yao Zhang<sup>*1</sup>,
> Jiaming Lv<sup>1</sup>,
> Jiangtao Xie<sup>1</sup>,
> Peihua Li<sup>1&#8224;</sup>
> </em>
> </p>
>
> **Abstract:** Few-shot action recognition (FSAR) has recently made notable progress through set matching and efficient adaptation of large-scale pre-trained models. However, two key limitations persist. First, existing set matching metrics typically rely on cosine similarity to measure inter-frame linear dependencies and then perform matching with only instance-level information, thus failing to capture more complex patterns such as nonlinear relationships and overlooking task-specific cues. Second, for efficient adaptation of CLIP to FSAR, recent work performing fine-tuning via skip-fusion layers (which we refer to as side layers) has significantly reduced memory cost. However, the newly introduced side layers are often difficult to optimize under limited data conditions. To address these limitations, we propose TS-FSAR, a framework comprising three components: (1) a visual Ladder Side Network (LSN) for efficient CLIP fine-tuning; (2) a metric called Task-Specific Distance Correlation Matching (TS-DCM), which uses α-distance correlation to model both linear and nonlinear inter-frame dependencies and leverages a task prototype to enable task-specific matching; and (3) a Guiding LSN with Adapted CLIP (GLAC) module, which regularizes LSN using the adapted frozen CLIP to improve training for better α-distance correlation estimation under limited supervision. Extensive experiments on five widely used benchmarks demonstrate that our TS-FSAR yields superior performance compared to prior state-of-the-arts.

## Project Structure

> Note: The `model/` directory contains large pre-trained CLIP weights and is not included in this repository. You can download it from [Baidu Cloud](https://pan.baidu.com/s/1De_AFTCbwaFnRpzwiikSrQ?pwd=278c) or [Google Drive](https://drive.google.com/drive/folders/1oSdrD10AU1kwuyPdVTz5tN87QVNn2jod?usp=drive_link). After downloading, place the vit-b-16 folder inside the `model/` directory.

```bash
TS-FSAR/
├── assets/             # Figures for README
├── configs/            # Configuration files for different datasets
├── datasets/           # Episodic sampling and data augmentation 
├── model/              # Pre-trained CLIP weights and config files (ViT-B/16)
├── prompts/            # LLM-generated class descriptions for different datasets
├── scripts/            # Shell scripts for training and testing
├── trainers/           # Core model architecture and method implementation
├── utils/              # Helper functions
├── main.py             # Entry point for training and evaluation 
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Implementations

### Environment Setup

> Hardware: Experiments were conducted using a single NVIDIA RTX 4090 GPU.  
> *Note: We strongly recommend aligning your hardware and software environment with ours to ensure reproducibility.*

Please create the same software environment as ours using the following commands :

```bash
conda create -n ts_fsar python=3.8
conda activate ts_fsar
pip install -r requirements.txt
```

### Data Preparation
>We support standard few-shot benchmarks including **SSv2-Full**, **SSv2-Small**, **Kinetics-100**, **HMDB51**, and **UCF101**.  
> ***Note:** We adopt the data preparation pipeline and split settings from [CLIP-FSAR](https://github.com/alibaba-mmai-research/CLIP-FSAR).
Please organize your datasets as follows :*

```bash
datapath/
├── smsm_full/
│   ├── 1.mp4
│   ├── 10.mp4
│   └── ...
├── smsm_small/
│   ├── 100001.mp4
│   └── ...
├── hmdb51/
│   └── videos/
│       ├── brush_hair/
│       │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi
│       │   └── ...
│       └── ... 
├── ucf101/
│   └── videos/
│       ├── ApplyEyeMakeup/
│       │   ├── v_ApplyEyeMakeup_g01_c01.avi
│       │   └── ...
│       └── ... 
├── kinetics100/
│   ├── air_drumming/
│   │   ├──-eGnPDG5Pxs_000053_000063.mp4.avi
│   │   └── ...
│   └── ... 
```

### Running

> Before executing the scripts, please update the `DATA_ROOT_DIR` field in the corresponding YAML files under `configs/` to match the path of your local datasets.  
Then you can run the code using the commands below :

```bash
# Take SSv2_Full 1-shot as an example:
# Training
cd scripts/SSv2_Full/
sh metatrain_full_1shot.sh

# Testing
cd scripts/SSv2_Full/
sh metatest_full_1shot.sh
```

## Results

>This table reports the model’s classification performance on several standard video action recognition datasets. The best checkpoints for all datasets can be downloaded directly from [Baidu Cloud](https://pan.baidu.com/s/1f2x2T7hptChLkAiyWJPLow?pwd=89ys) or [Google Drive](https://drive.google.com/drive/folders/13D_3JJrA5aJldsHKc-ldrOsRe-K4anW0?usp=drive_link).

| Dataset | 5-way 1-shot | 5-way 5-shot |
| :--- | :---: | :---: |
| **SSv2-Full** | **75.1** | **83.5** |
| **SSv2-Small** | **60.5** | **70.3** |
| **HMDB51** | **85.0** | **88.9** |
| **UCF101** | **98.7** | **99.3** |
| **Kinetics** | **96.3** | **96.6** |

## Citation
If you find this work useful, please consider citing:

```Bibtex
@inproceedings{ts-fsar2026,
  title     = {Task-Specific Distance Correlation Matching for Few-Shot Action Recognition},
  author    = {Fei Long and Yao Zhang and Jiaming Lv and Jiangtao Xie and Peihua Li},
  booktitle = {The Fortieth AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

## Acknowledgements

Our code is built upon [CLIP-FSAR](https://github.com/alibaba-mmai-research/CLIP-FSAR). We thank the authors for their open-source contribution. We also acknowledge the efforts of the [CLIP](https://github.com/openai/CLIP) team for providing the powerful pre-trained models.

## Contact
If you have any questions, please create an issue or contact Yao Zhang at zyemail@mail.dlut.edu.cn.

## DTO-KD gradient-coordination experiments

The branch `research/dto-kd-fsar-gradient-coordination` contains the CPU-validated DTO-KD adapter and portable campus scripts. It provides E0 (original), E1 (amortized DTO-KD), E2 (static KL controls at 0.05/0.1/0.2), and E3 (gradient-scale-matched KL) for SSv2-Small 1-shot and 5-shot. No formal GPU training is started on the development server.

See [`docs/CAMPUS_DEPLOYMENT.md`](docs/CAMPUS_DEPLOYMENT.md) for the complete deployment order. The code repository excludes videos, CLIP weights, and checkpoints; import those separately, verify their SHA256 values, then run `scripts/campus_preflight.sh` and `scripts/run_campus_experiment.sh`.
