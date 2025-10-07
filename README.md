# Lumina2-LoRA-trainer
建议保留`train_lumina_24GLora.ps1`中除数据集外的参数设定，否则可能发生意料外的bug
W.I.P


# Features
1. 使用[EQ-VAE](https://huggingface.co/Anzhc/MS-LC-EQ-D-VR_VAE)替换原来的Flux VAE，且修改VAE nn.conv2d的padding_mode为"reflect" `sd-scripts/library/flux_models.py`
2. 实现原始Lumina Image 2.0的双损失函数（原分辨率loss -> 原分辨率loss + 4倍下采样loss），代价是失去“differential output preservation”功能（懒） `sd-scripts/lumina_train_network.py`
3. 对于`timestep_sampling="nextdit_shift"`模式，修改了`sd-scripts/library/lumina_train_util.py get_lin_function`函数以支持1536px分辨率 (原始实现只支持到1024px)
   * <sub>*该模式下，将对不同分辨率的图像应用不同的`discrete_flow_shift`，这意味着更高分辨率的图像将会被更频繁地加上更高的噪声（分辨率越高，图像对噪声的耐受程度越高），因此不同分辨率的图片在相同的t下将保持相同的“信息量”*</sub>
4. TODO
