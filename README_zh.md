# 基于 EasyR1 复现 Qwen2.5-VL-3B GRPO 训练

[English](README.md) | [简体中文](README_zh.md)

本项目基于 [EasyR1](https://github.com/hiyouga/EasyR1)，复现
[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
在 [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k)
数据集上的 GRPO 强化学习训练流程。

当前配置已在一台配备 8 张 NVIDIA GeForce RTX 3090 24GB 的服务器上验证，
训练使用其中的 GPU 0-3。采用全参数 BF16 训练，并集成 vLLM rollout、FSDP、
规则奖励、WandB、本地日志、GPU 空闲等待和训练前后回答对比。

## 训练流程

对于每个问题，当前策略模型生成 4 个候选回答。奖励函数分别判断最终答案
是否正确，以及输出是否满足指定格式。GRPO 在同一个问题的候选回答组内
归一化奖励，不需要额外训练 critic 模型。KL 惩罚用于限制策略模型过快偏离
参考模型。

模型需要输出：

```text
<think>推理过程</think><answer>最终答案</answer>
```

主要配置：

- 模型：`Qwen/Qwen2.5-VL-3B-Instruct`
- 数据集：`hiyouga/geometry3k`
- 算法：GRPO
- 精度：BF16
- 微调方式：全参数微调
- 分布式策略：四张 GPU 上使用 FSDP
- 生成引擎：vLLM
- 奖励：答案准确率和输出格式
- 监控：本地 JSONL 日志和 Weights & Biases

## 环境要求

- Linux
- Python 3.9+
- 支持 BF16 的 NVIDIA GPU
- 与 PyTorch、FlashAttention、vLLM 兼容的 CUDA 环境
- 服务器：8 张 NVIDIA GeForce RTX 3090 24GB
- 训练设备：GPU 0-3（`CUDA_VISIBLE_DEVICES=0,1,2,3`）
- NVIDIA 驱动：530.30.02（`nvidia-smi` 显示 CUDA 12.1）
- Conda 环境：PyTorch 2.8.0 cu126、CUDA Toolkit 12.6

EasyR1 上游资源表估算 Qwen2.5-VL-3B 全参数 BF16 训练最低约需
`1 x 40GB`。实际显存峰值取决于图片尺寸、提示长度、生成长度和 rollout
参数。

基于 Python 3.10、PyTorch 2.8.0 cu126、vLLM 0.11.0 和
FlashAttention 2.8.3 的已验证 RTX 3090 环境配置参见
[RTX3090_ENVIRONMENT_SETUP_ZH.md](RTX3090_ENVIRONMENT_SETUP_ZH.md)。

## 安装

推荐使用 EasyR1 预构建镜像：

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all \
  -v "$PWD":/workspace/Qwen2.5-VL-3B-GRPO \
  hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

也可以在已有兼容环境中安装：

```bash
git clone https://github.com/k-irona/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1.git
cd Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1
pip install -e .
```

本仓库固定使用 vLLM 0.11.0。不要安装未固定版本的新版 vLLM，其内部 LoRA
API 已发生变化，与当前 EasyR1 代码不兼容。

无法直接访问 Hugging Face 时：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始

需要在线可视化时，先登录一次：

```bash
wandb login
```

使用四张 RTX 3090 运行 10 step 测试：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPUS=4 \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh \
trainer.max_steps=10
```

开始完整训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPUS=4 \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh
```

训练进程内部会将选中的物理 GPU 映射为逻辑设备 `0,1,2,3`。当前过滤后的
Geometry3K 训练 DataLoader 每个 epoch 包含 65 step，因此默认 15 epoch
约为 975 step。实际总步数应以启动日志中的 `Total training steps` 为准。

## 默认参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `N_GPUS` | 2 | 脚本默认值；当前 RTX 3090 方案设为 `4` |
| `ROLLOUT_BATCH_SIZE` | 32 | 每轮 rollout 使用的问题数 |
| `GLOBAL_BATCH_SIZE` | 16 | actor 更新 minibatch 大小 |
| `MICRO_BATCH_SIZE_UPDATE` | 1 | 每张 GPU 的 actor 更新 micro batch 大小 |
| `MICRO_BATCH_SIZE_EXPERIENCE` | 1 | 每张 GPU 计算 log-prob 的 micro batch 大小 |
| `ROLLOUT_N` | 4 | 每个问题生成的候选回答数 |
| `MAX_PROMPT_LENGTH` | 2048 | 最大提示 token 数 |
| `MAX_RESPONSE_LENGTH` | 512 | 最大生成 token 数 |
| `MAX_PIXELS` | 524288 | 单张输入图片最大像素数 |
| `GPU_MEMORY_UTILIZATION` | 0.5 | vLLM 可使用的显存比例 |
| `LEARNING_RATE` | `1e-6` | 全参数 actor 学习率 |
| `KL_COEF` | `1e-2` | KL 正则系数 |
| `LORA_RANK` | 0 | 0 表示关闭 LoRA，进行全参数训练 |
| `LORA_ALPHA` | 64 | 启用 LoRA 时的缩放系数 |
| `TOTAL_EPOCHS` | 15 | 总训练 epoch 数 |
| `VAL_FREQ` | 5 | 验证间隔 step 数 |
| `SAVE_FREQ` | 5 | checkpoint 保存间隔 |
| `LOGGER` | `['file','wandb']` | 启用的日志后端 |
| `EXPERIMENT_NAME` | `qwen2_5_vl_3b_geo_grpo_3090_bf16` | 实验名和 checkpoint 目录名 |

所有参数均可通过环境变量覆盖：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPUS=4 \
ROLLOUT_BATCH_SIZE=16 \
GLOBAL_BATCH_SIZE=8 \
MAX_RESPONSE_LENGTH=384 \
MAX_PIXELS=393216 \
GPU_MEMORY_UTILIZATION=0.45 \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh
```

也可以在命令末尾追加 OmegaConf 参数：

```bash
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh \
trainer.max_steps=100 \
trainer.val_freq=25
```

## 自动等待 GPU

等待脚本会监控 GPU 显存和利用率。当前方案会在同样的四张 GPU 连续三次
满足空闲条件后自动启动训练。

等待任意四张 GPU：

```bash
NUM_GPUS=4 \
nohup bash scripts/wait_for_gpus_and_train.sh \
  > wait_for_gpus.log 2>&1 &
```

只允许从指定 GPU 中选择：

```bash
GPU_CANDIDATES=4,5,6,7 \
NUM_GPUS=4 \
nohup bash scripts/wait_for_gpus_and_train.sh \
  > wait_for_gpus.log 2>&1 &
```

查看等待或训练日志：

```bash
tail -f wait_for_gpus.log
```

默认空闲条件：

- 已使用显存低于 2000 MiB
- GPU 利用率低于 10%
- 每 60 秒检查一次
- 连续满足 3 次

可通过 `MAX_MEMORY_USED_MB`、`MAX_GPU_UTILIZATION`、
`POLL_INTERVAL_SECONDS` 和 `REQUIRED_FREE_CHECKS` 修改条件。

该脚本只能检测空闲状态，不能阻止其他用户同时占用 GPU。如果实验室配置了
Slurm 等调度系统，应优先使用调度器。

## 训练可视化

### Weights & Biases

默认日志配置为：

```text
['file', 'wandb']
```

训练开始后访问 [wandb.ai](https://wandb.ai/)：

- Project：`easy_r1`
- Run：`qwen2_5_vl_3b_geo_grpo_3090_bf16`

建议关注：

- `val/accuracy_reward`
- `val/reward_score`
- `val/format_reward`
- `actor/ppo_kl`
- `actor/pg_loss`
- `response_length/clip_ratio`
- `val/generations`

训练会在 step 0 先验证基础模型，因此同一次 run 中已经包含训练前基线和训练
后的结果。

### 本地 HTML 报告

日志默认保存在：

```text
checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16/
```

生成本地报告：

```bash
python3 scripts/visualize_training.py \
  checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16
```

输出文件：

```text
checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16/training_report.html
```

报告包含指标曲线，以及 step 0 和最后一次验证时的回答、标签和得分对比。

只保留本地日志、不使用 WandB：

```bash
LOGGER="['file']" \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh
```

## Checkpoint

Checkpoint 保存在实验目录中。将 actor checkpoint 转换为 Hugging Face 格式：

```bash
python3 scripts/model_merger.py \
  --local_dir checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16/global_step_<STEP>/actor
```

当 `trainer.find_last_checkpoint=true` 时，训练器会自动从同一实验目录中的最新
checkpoint 恢复。

## 显存调整

发生 OOM 时，建议按以下顺序调整：

1. 设置 `MAX_PIXELS=393216`。
2. 设置 `MAX_RESPONSE_LENGTH=384`。
3. 设置 `GPU_MEMORY_UTILIZATION=0.45`。
4. 保持两个 micro batch size 均为 `1`。
5. 设置 `ROLLOUT_BATCH_SIZE=16 GLOBAL_BATCH_SIZE=8`。
6. 使用 `LORA_RANK=32 LEARNING_RATE=1e-5` 切换为 LoRA。

GRPO 要求 `ROLLOUT_N > 1`，不能将其降低为 1。

## 常见问题

### Image features and image tokens do not match

增大 `MAX_PROMPT_LENGTH`，或者降低 `MAX_PIXELS`。

### CUDA out of memory

降低 `GPU_MEMORY_UTILIZATION`、`MAX_PIXELS` 和
`MAX_RESPONSE_LENGTH`。基础配置已默认启用参数和优化器 CPU offload。

### 自动训练卡在 WandB 登录

在启动 GPU 等待脚本前运行 `wandb login`，或者使用：

```bash
LOGGER="['file']" bash scripts/wait_for_gpus_and_train.sh
```

### Ray 报告 no active drivers

按照 EasyR1 上游建议，移除环境中冲突的 DeepSpeed 安装。

## 主要文件

```text
examples/qwen2_5_vl_3b_geo3k_grpo.sh  GRPO 训练入口
examples/reward_function/r1v.py       准确率和格式奖励
examples/format_prompt/r1v.jinja      推理与答案格式模板
scripts/wait_for_gpus_and_train.sh    自动等待 GPU 并启动训练
scripts/visualize_training.py         本地 HTML 报告生成器
docs/qwen2_5_vl_3b_grpo_zh.md         补充调参说明
```

## 致谢

本项目基于 [EasyR1](https://github.com/hiyouga/EasyR1)，EasyR1 构建于
[veRL](https://github.com/volcengine/verl)。感谢 EasyR1、veRL、
Qwen2.5-VL、vLLM 和 Geometry3K 的作者与贡献者。

## 引用

如果将本项目用于研究，请引用 EasyR1 和 HybridFlow：

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong, Richong Zhang},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}

@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  journal = {arXiv preprint arXiv:2409.19256},
  year    = {2024}
}
```

## 许可证

本项目沿用上游 Apache License 2.0，参见 [LICENSE](LICENSE)。
