#!/bin/bash
set -e

# 设置临时目录到 NVMe 磁盘，避免 OOM
export TMPDIR=/opt/dlami/nvme/tmp
export TEMP=/opt/dlami/nvme/tmp
export TMP=/opt/dlami/nvme/tmp

# PyTorch 共享内存路径（多进程通信使用）
export TORCH_SHARED_MEMORY_DIR=/opt/dlami/nvme/tmp/torch_shm

# 设置 HuggingFace 相关缓存目录（统一使用同一个路径）
export HF_HOME=/opt/dlami/nvme/tmp/hf_cache
export HF_HUB_CACHE=/opt/dlami/nvme/tmp/hf_cache
export TRANSFORMERS_CACHE=/opt/dlami/nvme/tmp/hf_cache
export HF_DATASETS_CACHE=/opt/dlami/nvme/tmp/hf_cache
export HF_DATASETS_OFFLINE=0
# 确保使用本地缓存
export HF_HUB_OFFLINE=0

# 设置 PyTorch 临时目录
export TORCH_HOME=/opt/dlami/nvme/tmp/torch_cache

# 创建必要的目录
mkdir -p $TMPDIR
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p /opt/dlami/nvme/tmp/axolotl_prepared_cache
mkdir -p /opt/dlami/nvme/tmp/torch_shm

# 清理旧的 PyTorch 共享内存文件（如果存在）
echo "清理旧的临时文件..."
find /opt/dlami/nvme/tmp/torch_shm -name "torch_*" -type f -mtime +1 -delete 2>/dev/null || true

# 清理 /dev/shm 和 /tmp 中的 torch 相关文件
echo "清理 /dev/shm 中的 torch 文件..."
find /dev/shm -name "torch_*" -type f -delete 2>/dev/null || true
find /dev/shm -name "sem.torch*" -delete 2>/dev/null || true

echo "清理 /tmp 中的 torch 文件..."
find /tmp -name "torch_*" -type f -delete 2>/dev/null || true
find /tmp -name "tmp*" -type f -mtime +1 -delete 2>/dev/null || true

# 增加文件描述符限制
ulimit -n 65536 2>/dev/null || true

# 检查磁盘空间
echo "检查磁盘空间:"
df -h /opt/dlami/nvme/tmp | tail -1

# 预下载模型（避免多进程同时下载导致冲突）
# 使用 huggingface-cli 下载，更轻量且不会加载到内存
echo "检查/预下载模型 Qwen/Qwen2.5-7B..."
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download Qwen/Qwen2.5-7B --cache-dir $HF_HOME --local-dir-use-symlinks False || echo "模型可能已存在或下载失败，继续..."
else
    echo "huggingface-cli 未安装，跳过预下载（将在训练时自动下载）"
fi

# 设置其他可能需要的环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 禁用 PyTorch 的共享内存（使用文件系统，避免多进程通信问题）
export TORCH_SHARED_MEMORY_ALLOCATION=file_system

# 设置 DataLoader 相关环境变量（如果 axolotl 支持）
export DATALOADER_NUM_WORKERS=0
export DATALOADER_PIN_MEMORY=false

# 限制 PyTorch 的多进程共享内存使用
export PYTORCH_MULTIPROCESSING_SHARING_STRATEGY=file_descriptor

echo "临时目录设置:"
echo "  TMPDIR=$TMPDIR"
echo "  HF_HOME=$HF_HOME"
echo "  TORCH_HOME=$TORCH_HOME"
echo ""

# 运行 axolotl 训练
accelerate launch -m axolotl.cli.train qwen2-5_7b.yaml

