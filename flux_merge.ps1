# LoRA svd_merge script by @bdsqlsz

$save_precision = "" # precision in saving, default float | 保存精度, 可选 float、fp16、bf16, 默认 和源文件相同
$precision = "float" # precision in merging (float is recommended) | 合并时计算精度, 可选 float、fp16、bf16, 推荐float
$flux_model = "./sd-models/flux1-dev.safetensors" # flux model path | flux大模型路径，如果需要融合到大模型填写
$clip_l=""  #clip lora weight|clip_l模型路径,如果需要融合到大模型填写
$t5xxl="" #t5xxl weight|t5xxl模型路径,如果需要融合到大模型填写
$models = "./output/Hyper-FLUX.1-dev-8steps-lora.safetensors" # original LoRA model path need to resize, save as cpkt or safetensors | 需要合并的模型路径, 保存格式 cpkt 或 safetensors，多个用空格隔开
$ratios = "1.0" # ratios for each model / LoRA模型合并比例，数量等于模型数量，多个用空格隔开
$save_to = "./output/Hyper-FLUX-8steps-lora.safetensors" # output model path, save as ckpt or safetensors | 输出路径, 保存格式 cpkt 或 safetensors
$clip_l_save_to="" #clip lora model path|clip_l模型输出路径, 保存格式 cpkt 或 safetensors
$t5xxl_save_to="" #t5xxl model path|t5xxl模型输出路径, 保存格式 cpkt 或 safetensors
$loading_device = "cuda" # device to load, cpu for CPU | 加载设备, 默认 CPU
$working_device = "cuda" # device to use, cuda for GPU | 使用 GPU跑, 默认 CPU
$concat=0 #concat lora instead of merge (The dim(rank) of the output LoRA is the sum of the input dims)|截取而不是合并
$shuffle=0 #shuffle lora weight|打乱权重
$diffusers=1 #diffuse lora weight|使用diffusers的权重

# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()

[void]$ext_args.Add("--model")
foreach ($model in $models.Split(" ")) {
    [void]$ext_args.Add($model)
}

[void]$ext_args.Add("--ratios")
foreach ($ratio in $ratios.Split(" ")) {
    [void]$ext_args.Add([float]$ratio)
}

if ($flux_model) {
  [void]$ext_args.Add("--flux_model=" + $flux_model)
}

if ($clip_l) {
  [void]$ext_args.Add("--clip_l=" + $clip_l)
}

if ($t5xxl) {
  [void]$ext_args.Add("--t5xxl=" + $t5xxl)
}

if ($save_precision) {
  [void]$ext_args.Add("--save_precision=" + $save_precision)
}

if ($loading_device) {
  [void]$ext_args.Add("--loading_device=" + $loading_device)
}

if ($working_device) {
  [void]$ext_args.Add("--working_device=" + $working_device)
}

if ($clip_l_save_to) {
  [void]$ext_args.Add("--clip_l_save_to=" + $clip_l_save_to)
}

if ($t5xxl_save_to) {
  [void]$ext_args.Add("--t5xxl_save_to=" + $t5xxl_save_to)
}

if ($concat) {
  [void]$ext_args.Add("--concat")
}

if ($shuffle) {
  [void]$ext_args.Add("--shuffle")
}

if ($diffusers) {
  [void]$ext_args.Add("--diffusers")
}

# run svd_merge
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/networks/flux_merge_lora.py" `
	--precision=$precision `
	--save_to=$save_to `
	$ext_args 

Write-Output "SVD Merge finished"
Read-Host | Out-Null ;
