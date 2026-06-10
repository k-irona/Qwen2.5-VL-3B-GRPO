# EasyR1 GRPO 环境配置交接说明

## 目标

为本仓库配置 Qwen2.5-VL-3B GRPO 训练环境。

仓库路径：

```text
/data/jinda/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1
```

Conda 环境：

```text
/data/jinda/envs/qwen2.5vl-grpo
```

## 已确认的硬件

- 当前机器有 8 张 GPU 可见。
- 当前计划使用 4 张 RTX 3090 24GB。
- RTX 3090 属于 Ampere，计算能力为 8.6。
- 训练建议先使用 4 卡全参数 BF16 做 10 step 测试；如果显存不足，再切换 LoRA。

## 当前 Python 环境

已确认：

```text
Python: 3.10
PyTorch: 2.8.0+cu126
PyTorch CUDA runtime: 12.6
vLLM: 0.11.0
torch.cuda.is_available(): True
torch.cuda.device_count(): 8
```

检查命令：

```bash
python -c "import torch,vllm; print('torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('vLLM:',vllm.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPUs:',torch.cuda.device_count())"
```

## CUDA Toolkit 状态

Conda 环境内已经安装 CUDA 12.6 编译器：

```text
nvcc: release 12.6, V12.6.85
```

应确保：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
hash -r
```

检查：

```bash
which nvcc
nvcc --version
```

预期：

```text
/data/jinda/envs/qwen2.5vl-grpo/bin/nvcc
release 12.6
```

## FlashAttention 当前状态

目标版本：

```text
flash-attn==2.8.3
```

FlashAttention 目前尚未成功安装。

### 已排除的问题

1. 旧 wheel 不可使用：

   ```text
   flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE
   ```

   它只适用于 PyTorch 2.5 和 ABI FALSE。

2. 官方/第三方预编译 FlashAttention 2.8.3 wheel 导入失败：

   ```text
   GLIBC_2.32 not found
   ```

   说明服务器 glibc 较旧，因此需要在本机源码编译。

3. 最初源码编译使用 Conda 编译器时，nvcc 收到重复 `-ccbin`，报错：

   ```text
   nvcc fatal: A single input file is required for a non-link phase
   ```

   已改为使用：

   ```bash
   CC=/usr/bin/gcc
   CXX=/usr/bin/g++
   CUDAHOSTCXX=/usr/bin/g++
   ```

4. 当前最新错误：

   ```text
   fatal error: cuda_bf16.h: No such file or directory
   ```

   文件实际存在：

   ```text
   $CONDA_PREFIX/targets/x86_64-linux/include/cuda_bf16.h
   $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/include/cuda_bf16.h
   ```

   问题是 C++ 编译命令没有包含
   `$CONDA_PREFIX/targets/x86_64-linux/include`。

## 下一步：重新编译 FlashAttention

先激活环境：

```bash
conda activate /data/jinda/envs/qwen2.5vl-grpo
```

设置环境变量：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export C_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
hash -r
```

确认头文件和编译器：

```bash
test -f "$CONDA_PREFIX/targets/x86_64-linux/include/cuda_bf16.h" && echo "cuda_bf16.h OK"
which nvcc
nvcc --version
/usr/bin/g++ --version
```

清理旧安装：

```bash
pip uninstall -y flash-attn
```

建议使用显式 include 参数重新编译：

```bash
CFLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include" \
CXXFLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include" \
FLASH_ATTN_CUDA_ARCHS=80 \
MAX_JOBS=2 \
NVCC_THREADS=2 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
pip install flash-attn==2.8.3 \
  --no-build-isolation \
  --no-cache-dir
```

注意：

- FlashAttention 的 Ampere 构建使用 `FLASH_ATTN_CUDA_ARCHS=80`。
- RTX 3090 的计算能力虽然是 8.6，但这里不应改成 `86`。
- 编译输出中的 `/usr/bin/g++` 命令必须出现：

  ```text
  -I/data/jinda/envs/qwen2.5vl-grpo/targets/x86_64-linux/include
  ```

如果仍找不到头文件，可以将单个头文件链接到 Conda include：

```bash
ln -sf \
  "$CONDA_PREFIX/targets/x86_64-linux/include/cuda_bf16.h" \
  "$CONDA_PREFIX/include/cuda_bf16.h"
```

然后重试编译。

## FlashAttention 安装后验证

```bash
python -c "import torch,flash_attn; print('torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('FlashAttention:',flash_attn.__version__)"
```

完整检查：

```bash
python -c "import torch,vllm,flash_attn; from vllm.lora.models import LoRAModel; print('torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('vLLM:',vllm.__version__); print('FlashAttention:',flash_attn.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPUs:',torch.cuda.device_count()); print('Environment OK')"
```

## 项目安装

项目 `requirements.txt` 已固定：

```text
vllm==0.11.0
```

为了避免 pip 重新替换当前 PyTorch/vLLM，使用：

```bash
cd /data/jinda/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1
pip install -e . --no-deps
```

## 4×RTX 3090 训练测试

完成环境验证后，先执行 10 step 测试。以下假设使用 GPU 0、1、2、3：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPUS=4 \
LORA_RANK=0 \
ROLLOUT_BATCH_SIZE=32 \
GLOBAL_BATCH_SIZE=16 \
ROLLOUT_N=4 \
MAX_RESPONSE_LENGTH=512 \
MAX_PIXELS=524288 \
GPU_MEMORY_UTILIZATION=0.4 \
LEARNING_RATE=1e-6 \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh \
trainer.max_steps=10
```

如果发生 OOM，先改为：

```text
MAX_RESPONSE_LENGTH=384
MAX_PIXELS=393216
GPU_MEMORY_UTILIZATION=0.35
ROLLOUT_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=8
```

如果仍然 OOM，切换 LoRA：

```text
LORA_RANK=32
LEARNING_RATE=1e-5
```

## 自动等待 GPU

测试成功后，可等待 4 张 GPU 空闲：

```bash
NUM_GPUS=4 \
GPU_CANDIDATES=0,1,2,3,4,5,6,7 \
nohup bash scripts/wait_for_gpus_and_train.sh \
  > wait_for_gpus.log 2>&1 &
```

查看日志：

```bash
tail -f wait_for_gpus.log
```

## WandB

启动无人值守训练前先执行：

```bash
wandb login
```

默认项目和实验名称：

```text
Project: easy_r1
Run: qwen2_5_vl_3b_geo_grpo_a40_bf16
```

虽然实验名中仍包含 `a40`，但不影响 RTX 3090 训练；后续可以通过
`EXPERIMENT_NAME` 环境变量覆盖。
