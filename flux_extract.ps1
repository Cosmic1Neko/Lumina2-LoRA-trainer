# LoRA svd_merge script by @bdsqlsz

$save_precision = "" # precision in saving, default float | 保存精度, 可选 float、fp16、bf16, 默认 和源文件相同
$model_org = "./sd-models/flux1-dev.safetensors" # flux model path | flux大模型路径，如果需要融合到大模型填写
$model_tuned = "./output/Hyper-FLUX-8steps-lora.safetensors" # original LoRA model path need to resize, save as cpkt or safetensors | 需要合并的模型路径, 保存格式 cpkt 或 safetensors，多个用空格隔开
$dim = 1 # dim to merge | 合并的维度
$save_to = "./output/Hyper-FLUX.1-dev-8steps-lora_rank1.safetensors" # output LoRA model path, save as ckpt or safetensors | 输出路径, 保存格式 cpkt 或 safetensors
$device = "cuda" # device to load, cpu for CPU | 加载设备, 默认 CPU
$clamp_quantile = 0.99 # clamp quantile | 量化的量化范围
$mem_eff_safe_open = 1 # use safe_open for memory efficiency | 使用safe_open提高内存效率

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

if ($save_precision) {
  [void]$ext_args.Add("--save_precision=" + $save_precision)
}

if ($dim) {
  [void]$ext_args.Add("--dim=" + $dim)
}

if ($device) {
  [void]$ext_args.Add("--device=" + $device)
}

if ($mem_eff_safe_open) {
  [void]$ext_args.Add("--mem_eff_safe_open")
}

if ($clamp_quantile -ne 0.99) {
  [void]$ext_args.Add("--clamp_quantile=" + $clamp_quantile)
}

# run svd_merge
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/networks/flux_extract_lora.py" `
  --model_org=$model_org `
  --model_tuned=$model_tuned `
	--save_to=$save_to `
	$ext_args 

Write-Output "SVD Merge finished"
Read-Host | Out-Null ;
