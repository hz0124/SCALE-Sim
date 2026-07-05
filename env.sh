# 激活 conda 环境
source /share/liujun/apps/anaconda3/bin/activate zhb-scalesim-v3
source /share/liujun/xinhaolee/vpn/proxy_infini.sh

export HF_HUB_DOWNLOAD_TIMEOUT=10
export HF_HOME=/share/liujun/.cache/huggingface
export HF_DATASETS_CACHE="/share/liujun/.cache/huggingface/hub"
export HF_ENDPOINT=https://hf-mirror.com

# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0

export GIT_AUTHOR_NAME='Zhh'
export GIT_AUTHOR_EMAIL='xinhaolee962@gmail.com'