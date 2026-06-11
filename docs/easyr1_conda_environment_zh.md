# 使用 Conda 配置 EasyR1 训练环境

本文记录 EasyR1 在 RTX 3090 集群上的 Conda 环境配置流程。完成环境配置后，
实际实验过程与指标解读参见
[《Qwen2.5-VL-3B GRPO 训练分析》](qwen2_5_vl_3b_grpo_training_analysis_zh.md)。

### 使用集群
NVIDIA GeForce RTX 3090 8x，训练使用 GPU 0-3
-  GPU memory：24GB per GPU
-  GPU architecture：Ampere
-  Compute capability：8.6
-  NVIDIA Driver Version：530.30.02
-  `nvidia-smi` CUDA Version：12.1

本文实际使用 4 张 RTX 3090、PyTorch 2.8.0 cu126、Conda CUDA Toolkit 12.6、vLLM 0.11.0 和 FlashAttention 2.8.3。
`nvidia-smi` 显示的 CUDA 12.1 表示驱动报告的 CUDA 兼容能力，不是当前 Conda 环境中 PyTorch runtime 或 `nvcc` 的版本；后两者均为 12.6。驱动 530.30.02 满足 CUDA 12.x minor version compatibility 的最低要求。

检查集群：
```
nvidia-smi
ldd --version | head -1
df -h /tmp
df -h /data
```
### 创建环境
Python Version：3.10
```
export WORK_ROOT=/data/<username>
export REPO_DIR=$WORK_ROOT/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1
export ENV_DIR=$WORK_ROOT/envs/qwen2.5vl-grpo
export MODEL_DIR=$WORK_ROOT/models/Qwen2.5-VL-3B-Instruct
export HF_HOME=$WORK_ROOT/huggingface
git clone https://github.com/k-irona/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1.git \
"$REPO_DIR"
conda create -p "$ENV_DIR" python=3.10 pip -y
conda activate "$ENV_DIR"
pip install -U pip setuptools wheel ninja packaging
```
### 安装 PyTorch
安装已经验证的 PyTorch 2.8.0 cu126：
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
--index-url https://download.pytorch.org/whl/cu126
```
验证：
```
python -c "import torch; print('torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('available:',torch.cuda.is_available()); print('GPUs:',torch.cuda.device_count())"
```
预期：
```
torch: 2.8.0+cu126
CUDA: 12.6
available: True
```
### 安装 CUDA Toolkit
FlashAttention 源码编译不仅需要安装 PyTorch，还需要安装 CUDA Toolkit，一定要注意 **nvcc 与 PyTorch CUDA 版本保持一致**，否则很容易出现兼容问题。
这里容易产生一个误区：虽然 `torch==2.8.0+cu126` 已经包含 CUDA Runtime，可以正常调用 GPU 进行训练，但 FlashAttention 需要在本机编译 CUDA 扩展，因此还需要额外提供：
-  CUDA 编译器（`nvcc`）
-  CUDA 头文件（如 `cuda_runtime.h`、`cuda_bf16.h`）
-  CUDA 开发库

这些组件由 CUDA Toolkit 提供，而不是 PyTorch wheel 提供。
因此，即使 PyTorch 已经能够正常识别 GPU，源码编译 FlashAttention 时仍然需要单独安装 CUDA Toolkit。
```
conda install -y -c nvidia cuda-toolkit=12.6
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$PATH"
hash -r
which nvcc
nvcc --version
```
如果没有 nvcc：
```
conda install -y -c nvidia cuda-nvcc=12.6
```
不要将 `$CONDA_PREFIX/lib` 永久加入 `LD_LIBRARY_PATH`，否则可能导致系统 `curl`、`bash` 出现 `libffi` 或 `libtinfo` 错误。
### 安装其他依赖
```
cd "$REPO_DIR"
pip install vllm==0.11.0 hydra-core \
-i https://pypi.tuna.tsinghua.edu.cn/simple
grep -v '^flash-attn' requirements.txt \
> "$WORK_ROOT/requirements-no-flash.txt"
pip install -r "$WORK_ROOT/requirements-no-flash.txt" \
-i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e . --no-deps
```
验证关键版本：
```
python -c "import torch,vllm,transformers,verl; print(torch.__version__); print(vllm.__version__); print(transformers.__version__); print('EasyR1 import OK')"
```
要求：
```
torch==2.8.0+cu126
vllm==0.11.0
transformers>=4.54.0,<5.0.0
```
### 安装 FlashAttention
最后我们来安装最麻烦的 FlashAttention，若遇到问题，可以参考以下 FlashAttention 编译失败案例表：

| 报错                            | 常见原因                      | 解决方式                         |
| ----------------------------- | ------------------------- | ---------------------------- |
| `GLIBC_2.32 not found`        | wheel 依赖较新 glibc          | 改源码编译                        |
| `cuda_bf16.h: No such file`   | CUDA include 路径没找到        | 设置 `CUDA_INCLUDE` / `CPATH`  |
| `No space left on device`     | `/tmp` 空间不足               | 设置 `TMPDIR=/data/...`        |
| `ninja: build stopped`        | 编译并行过高/OOM                | 降低 `MAX_JOBS`                |
| `unsupported architecture 86` | flash-attn 2.8.3 不接受 `86` | 用 `FLASH_ATTN_CUDA_ARCHS=80` |
目标版本：`flash-attn==2.8.3`
预编译 wheel 安装速度最快，无需本地编译 CUDA 扩展，因此优先尝试 wheel 安装。
只有当 glibc、ABI 或平台兼容性问题导致 wheel 无法使用时，再退回源码编译。
先尝试官方预编译 wheel：
```
ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
FLASH_WHEEL="flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl"
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_WHEEL}"
mkdir -p "$WORK_ROOT/wheels"
env -u LD_LIBRARY_PATH /usr/bin/curl -L "$FLASH_URL" \
-o "$WORK_ROOT/wheels/$FLASH_WHEEL"
pip install "$WORK_ROOT/wheels/$FLASH_WHEEL"
python -c "import flash_attn; print(flash_attn.__version__)"
```
由于 wheel 依赖较新的 glibc，如果出现 `GLIBC_2.32 not found`、未定义符号或 wheel 不存在，则需要在本机源码编译：
```
pip uninstall -y flash-attn
export BUILD_TMP=$WORK_ROOT/tmp/flash-attn
export CUDA_INCLUDE="$CONDA_PREFIX/targets/x86_64-linux/include"
export TMPDIR="$BUILD_TMP"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export CPATH="$CUDA_INCLUDE:${CPATH:-}"
export C_INCLUDE_PATH="$CUDA_INCLUDE:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_INCLUDE:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
mkdir -p "$BUILD_TMP"
test -f "$CUDA_INCLUDE/cuda_bf16.h" && echo "cuda_bf16.h OK"
CFLAGS="-I$CUDA_INCLUDE" \
CXXFLAGS="-I$CUDA_INCLUDE" \
FLASH_ATTN_CUDA_ARCHS=80 \
MAX_JOBS=32 \
NVCC_THREADS=2 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
TMPDIR="$BUILD_TMP" \
pip install flash-attn==2.8.3 \
--no-build-isolation \
--no-cache-dir
```

RTX 3090 的 compute capability 是 8.6，但 FlashAttention 2.8.3 的 `FLASH_ATTN_CUDA_ARCHS` 构建选项并不接受 `86`。
该版本在官方 `setup.py` 中将 Ampere 内核统一配置为 `80`，因此这里应使用 `FLASH_ATTN_CUDA_ARCHS=80`；生成的 `sm_80` 内核可在 RTX 3090 这类 `sm_86` Ampere GPU 上运行。
源码编译建议准备 **至少 30GB 临时空间**，不要使用空间不足的系统 `/tmp`。编译过程中会生成大量 CUDA 中间文件（`.o`、PTX、fatbin 等），磁盘占用峰值可能达到数十 GB。
可以开另一个终端查看编译进度：
```
watch -n 2 "
echo 'Processes:'
pgrep -af 'ninja|nvcc' | wc -l
echo 'Object files:'
find '$BUILD_TMP' -name '*.o' 2>/dev/null | wc -l
echo 'Storage:'
du -sh '$BUILD_TMP' 2>/dev/null
"
```
`MAX_JOBS` 可根据机器 CPU 核数和可用内存调整。如果编译过程中出现 OOM、`ninja: build stopped` 或系统负载过高，可尝试降低至 8 或 16。
编译完成后验证：
```
python -c "import torch,flash_attn; print(torch.__version__); print(torch.version.cuda); print(flash_attn.__version__)"
rm -rf "$BUILD_TMP"
```
如果能够正常导入 `flash_attn` 且未出现 undefined symbol、GLIBC 或 CUDA Runtime 相关错误，则说明安装成功。
### 完整环境验证
```
pip check
python -c "import torch,vllm,flash_attn,transformers,verl; from vllm.lora.models import LoRAModel; print('torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('vLLM:',vllm.__version__); print('FlashAttention:',flash_attn.__version__); print('Transformers:',transformers.__version__); print('GPUs:',torch.cuda.device_count()); print('Environment OK')"
```
### 下载模型和数据集
```
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/<username>/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME" "$MODEL_DIR"
hf download Qwen/Qwen2.5-VL-3B-Instruct \
--local-dir "$MODEL_DIR"
```
下载 Geometry3K：
```
python -c "from datasets import load_dataset; print(load_dataset('hiyouga/geometry3k'))"
```
如果 `curl` 出现 `libffi` 符号错误：
```
unset LD_LIBRARY_PATH
env -u LD_LIBRARY_PATH /usr/bin/curl -I \
https://hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/config.json
```
### 配置 WandB
```
wandb login
export WANDB_ENTITY=<wandb-entity>
export WANDB_PROJECT_NAME="Qwen2.5-VL-3B GRPO Reproduction with EasyR1"
```
WandB 项目地址格式：
```
https://wandb.ai/<wandb-entity>/<project-name>
```
必须使用 URL 中的 entity，登录时显示的昵称不一定是 entity。
### 运行 10 Step 测试
先停止残留 Ray：
```
ray stop --force
unset LD_LIBRARY_PATH
```
4 张 RTX 3090：
```
WANDB_ENTITY="$WANDB_ENTITY" \
HF_ENDPOINT="$HF_ENDPOINT" \
HF_HOME="$HF_HOME" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH="$MODEL_DIR" \
N_GPUS=4 \
EXPERIMENT_NAME=qwen2_5_vl_3b_geo_grpo_3090_smoke10 \
LOGGER="['file','wandb']" \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh \
trainer.project_name="'$WANDB_PROJECT_NAME'" \
trainer.max_steps=10
```
默认参数：
```
ROLLOUT_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=16
MICRO_BATCH_SIZE_UPDATE=1
MICRO_BATCH_SIZE_EXPERIENCE=1
ROLLOUT_N=4
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=512
MAX_PIXELS=524288
LEARNING_RATE=1e-6
KL_COEF=1e-2
LORA_RANK=0
GPU_MEMORY_UTILIZATION=0.5
TOTAL_EPOCHS=15
```
### 查看训练详情
```
export RUN_DIR="$REPO_DIR/checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_smoke10"
tail -f "$RUN_DIR/experiment_log.jsonl"
```
查看最新一步：
```
tail -1 "$RUN_DIR/experiment_log.jsonl" | python -m json.tool
```
查看模型生成：
```
tail -f "$RUN_DIR/generations.log"
```
查看 GPU：
```
watch -n 2 nvidia-smi
```
Ray Dashboard：
```
ssh -L 8265:127.0.0.1:8265 <username>@<cluster-host>
```
浏览器访问：
```
http://127.0.0.1:8265
```
### 显存不足
依次降低：
```
MAX_PIXELS=393216
MAX_RESPONSE_LENGTH=384
GPU_MEMORY_UTILIZATION=0.45
ROLLOUT_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=8
```
仍然 OOM 时切换 LoRA：
```
LORA_RANK=32
LEARNING_RATE=1e-5
```
### 正式训练
10 step 测试通过后，移除 `trainer.max_steps=10`。当前 Geometry3K 过滤后训练 DataLoader 为 65 step/epoch，默认训练 15 epoch，因此约为 `65 × 15 = 975 step`；实际总步数以启动日志中的 `Total training steps` 为准。
```
ray stop --force
unset LD_LIBRARY_PATH
WANDB_ENTITY="$WANDB_ENTITY" \
HF_ENDPOINT="$HF_ENDPOINT" \
HF_HOME="$HF_HOME" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH="$MODEL_DIR" \
N_GPUS=4 \
EXPERIMENT_NAME=qwen2_5_vl_3b_geo_grpo_3090_full \
LOGGER="['file','wandb']" \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh \
trainer.project_name="'$WANDB_PROJECT_NAME'"
```

### 参考资料
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [CUDA 12.6 Release Notes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)
