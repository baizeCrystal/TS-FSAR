# 预训练权重与离线资产

TS-FSAR 实际通过 `transformers` 的 `CLIPVisionModelWithProjection`、`CLIPTextModelWithProjection` 和 `CLIPTokenizer` 从 `./model/vit-b-16/` 加载 **OpenAI CLIP ViT-B/16**。代码位置是 `trainers/model.py::build_model`，不是随机初始化，也不会自动切换到其他 CLIP 版本。

当前原服务器目录大小约 575 MB；逐文件 SHA256 在 `assets/CLIP_VIT_B16_SHA256SUMS.json`。其中 `pytorch_model.bin` 为 598,641,023 bytes，SHA256 为 `ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0`。权重来源记录在项目 README 的官方 Baidu Cloud/Google Drive 链接；本项目保留的是已持有的本地 Transformers 格式文件。

联网时可尝试：

```bash
python scripts/download_pretrained.py --repo-id openai/clip-vit-base-patch16
python scripts/verify_weights.py
```

下载后哈希不一致必须停止，不能静默替换。校园端无外网时，把 `CLIP-vit-b-16-v2.tar.gz` 解压得到的 `vit-b-16/` 通过用户电脑或平台文件区传入，再运行：

```bash
bash scripts/import_offline_assets.sh /path/to/offline_assets
python scripts/verify_weights.py
```

四个复现 checkpoint 只作为 Release/离线大文件资产，不提交普通 Git 历史；其大小和 SHA256 见 `assets/DTO_KD_CHECKPOINT_MANIFEST.json`。
