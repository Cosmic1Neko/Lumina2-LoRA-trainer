Set-Location $PSScriptRoot

$Env:HF_HOME="huggingface"
$Env:HF_ENDPOINT="https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK=1
$Env:PIP_NO_CACHE_DIR=1
#$Env:PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
$Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
$Env:UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"
$Env:UV_CACHE_DIR="${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_CACHE=0
$Env:UV_LINK_MODE="symlink"
$uv="~/.local/bin/uv"

function InstallFail {
    Write-Output "安装失败。"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

function Move-FileSafe {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    try {
        $destDir = Split-Path -Path $Destination -Parent
        if (-not (Test-Path -Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Move-Item -Force -Path $Source -Destination $Destination -ErrorAction Stop
        Write-Host "已重命名: $Source -> $Destination"
        return $true
    }
    catch {
        Write-Warning "重命名失败：$($_.Exception.Message)"
        return $false
    }
}

try {
    & $uv -V
    Write-Output "uv installed|UV模块已安装."
}
catch {
    Write-Output "Install uv|安装uv模块中..."
    if ($Env:OS -ilike "*windows*") {
        powershell -ExecutionPolicy ByPass -c "./uv-installer.ps1"
        Check "安装uv模块失败。"
    }
    else {
        sh "./uv-installer.sh"
        Check "安装uv模块失败。"
    }
}

if ($env:OS -ilike "*windows*") {
    #chcp 65001
    # First check if UV cache directory already exists
    if (Test-Path -Path "${env:LOCALAPPDATA}/uv/cache") {
        Write-Host "UV cache directory already exists, skipping disk space check"
    }
    else {
        # Check C drive free space with error handling
        try {
            $CDrive = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction Stop
            if ($CDrive) {
                $FreeSpaceGB = [math]::Round($CDrive.FreeSpace / 1GB, 2)
                Write-Host "C: drive free space: ${FreeSpaceGB}GB"
                
                # $Env:UV cache directory based on available space
                if ($FreeSpaceGB -lt 20) {
                    Write-Host "Low disk space detected. Using local .cache directory"
                    $Env:UV_CACHE_DIR = ".cache"
                } 
            }
            else {
                Write-Warning "C: drive not found. Using local .cache directory"
                $Env:UV_CACHE_DIR = ".cache"
            }
        }
        catch {
            Write-Warning "Failed to check disk space: $_. Using local .cache directory"
            $Env:UV_CACHE_DIR = ".cache"
        }
    }
    if (Test-Path "./venv/Scripts/activate") {
        Write-Output "Windows venv"
        . ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        Write-Output "Windows .venv"
        . ./.venv/Scripts/activate
    }
    else {
        Write-Output "Create .venv"
        & $uv venv -p 3.11 --seed
        . ./.venv/Scripts/activate
    }
}
elseif (Test-Path "./venv/bin/activate") {
    Write-Output "Linux venv"
    . ./venv/bin/activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
    Write-Output "Linux .venv"
    . ./.venv/bin/activate.ps1
}
else{
    Write-Output "Create .venv"
    & $uv venv -p 3.11 --seed
    . ./.venv/bin/activate.ps1
}

Set-Location .\sd-scripts
Write-Output "安装程序所需依赖 (已进行国内加速，若在国外或无法使用加速源请换用 install.ps1 脚本)"

& $uv pip install -U hatchling editables torch==2.8.0
Check "torch安装失败。"

& $uv pip sync ./requirements-uv.txt --index-strategy unsafe-best-match
Check "环境安装失败。"

& $uv pip install -U --pre lycoris-lora -i https://pypi.org/simple torch==2.8.0
Check "lycoris-lora安装失败。"

Set-Location ../
$download_fluxdev2pro = Read-Host "是否下载flux-SRPO模型? 若需要下载模型选择 y ，若不需要选择 n。[y/n] (默认为 n)"
if ($download_fluxdev2pro -eq "y" -or $download_fluxdev2pro -eq "Y"){
    if (-not (Test-Path "./Stable-diffusion/flux-SRPO.safetensors")) {
        huggingface-cli download tencent/SRPO diffusion_pytorch_model.safetensors --local-dir Stable-diffusion
        Check "模型下载失败。"

        # Rename downloaded file to flux-SRPO.safetensors (robust recursive search)
        $destDir = "Stable-diffusion"
        $dest = Join-Path $destDir "flux-SRPO.safetensors"
        try {
            $found = Get-ChildItem -Path $destDir -Recurse -Filter "diffusion_pytorch_model.safetensors" -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -ne $found) {
                $null = Move-FileSafe -Source $found.FullName -Destination $dest
            } else {
                Write-Warning "未在 $destDir 下找到 diffusion_pytorch_model.safetensors；请检查下载结构。"
            }
        } catch {
            Write-Warning "重命名失败：$($_.Exception.Message)"
        }
    }
}

$download_hy = Read-Host "是否下载hunyuanimage-2.1模型? 若需要下载模型选择 y ，若不需要选择 n。[y/n] (默认为 n)"
if ($download_hy -eq "y" -or $download_hy -eq "Y"){
    if (-not (Test-Path "./Stable-diffusion/HunyuanImage-2.1/dit/hunyuanimage2.1.safetensors")) {
        huggingface-cli download tencent/HunyuanImage-2.1 dit/hunyuanimage2.1.safetensors --local-dir Stable-diffusion
        Check "hunyuanimage2.1 下载失败。"
    }
    if (-not (Test-Path "./Stable-diffusion/split_files/text_encoders/qwen_2.5_vl_7b.safetensors")) {
        huggingface-cli download Comfy-Org/HunyuanImage_2.1_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b.safetensors --local-dir Stable-diffusion
        Check "qwen_2.5_vl_7b 下载失败。"
    }
    if (-not (Test-Path "./Stable-diffusion/split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors")) {
        huggingface-cli download Comfy-Org/HunyuanImage_2.1_ComfyUI split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors --local-dir Stable-diffusion
        Check "byt5_small_glyphxl_fp16 下载失败。"
    }
    if (-not (Test-Path "./Stable-diffusion/split_files/vae/hunyuan_image_2.1_vae_fp16.safetensors")) {
        huggingface-cli download Comfy-Org/HunyuanImage_2.1_ComfyUI split_files/vae/hunyuan_image_2.1_vae_fp16.safetensors --local-dir Stable-diffusion
        Check "vae 下载失败。"
    }
}

Write-Output "安装完毕"
Read-Host | Out-Null ;
