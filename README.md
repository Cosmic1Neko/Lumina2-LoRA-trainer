# Lumina2-LoRA-trainer
建议保留`train_lumina_24GLora.ps1`中除数据集外的参数设定，否则可能发生意料外的bug<br>
[Source](https://space.bilibili.com/219296)

# Features
1. 使用[EQ-VAE](https://huggingface.co/Anzhc/MS-LC-EQ-D-VR_VAE)替换原来的Flux VAE，且修改VAE nn.conv2d的padding_mode为"reflect" https://github.com/Cosmic1Neko/Lumina2-LoRA-trainer/blob/d301b7fbffe1c3b3025a4c4713d6ac93edf8ab6a/sd-scripts/library/flux_models.py#L326-L330
2. 实现原始Lumina Image 2.0的双损失函数（原分辨率loss -> 原分辨率loss + 4倍下采样loss），代价是失去“differential output preservation”功能 https://github.com/Cosmic1Neko/Lumina2-LoRA-trainer/blob/d578e5778b5b960f97f06c60b91b8b58a2095915/sd-scripts/lumina_train_network.py#L436-L437
3. 修改源代码使得gemma2_max_token_length最大支持数300 -> 1280 + 50 https://github.com/Cosmic1Neko/Lumina2-LoRA-trainer/blob/2e410fb1c5406264dc1aebb615579341fbf896a9/sd-scripts/library/lumina_models.py#L143

