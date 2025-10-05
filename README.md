# Lumina2-LoRA-trainer
W.I.P


# Features
1. 使用[EQ-VAE](https://huggingface.co/Anzhc/MS-LC-EQ-D-VR_VAE)替换原来的Flux VAE，且修改VAE nn.conv2d的padding_mode为"reflect" `sd-scripts/library/flux_models.py`
2. 实现原始Lumina Image 2.0的双损失函数（原分辨率loss -> 原分辨率loss + 4倍下采样loss） `sd-scripts/lumina_train_network.py`
