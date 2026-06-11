# Qwen2.5-VL-3B GRPO 训练分析

本文记录 Qwen2.5-VL-3B-Instruct 在 Geometry3K 上进行 GRPO 训练的实际过程，主要说明训练参数、GPU 故障后的断点续训，以及 3 epoch 训练结束后的结果。环境安装过程参见 [《使用 Conda 配置 EasyR1 训练环境》](easyr1_conda_environment_zh.md)。

## 1. 训练设置

实验使用 EasyR1，在 4 张 RTX 3090 24GB 上进行 BF16 全参数训练。数据集为 Geometry3K，使用 R1V 提示模板和规则奖励函数，其中 `format reward` 判断输出格式，`accuracy reward` 判断最终答案。每个问题生成 4 条回答，并在组内计算相对优势，不额外训练 critic 模型。

| 参数 | 实际值 |
| --- | ---: |
| GPU | 4 张 RTX 3090，设备 0-3 |
| `ROLLOUT_BATCH_SIZE` | 32 |
| `ROLLOUT_N` | 4 |
| `GLOBAL_BATCH_SIZE` | 16 |
| `MICRO_BATCH_SIZE_UPDATE` | 1 |
| `MICRO_BATCH_SIZE_EXPERIENCE` | 1 |
| `MAX_PROMPT_LENGTH` | 2048 |
| `MAX_RESPONSE_LENGTH` | 512 |
| `MAX_PIXELS` | 524288 |
| `GPU_MEMORY_UTILIZATION` | 0.5 |
| `LEARNING_RATE` | `1e-6` |
| `KL_COEF` | `1e-2` |
| `LORA_RANK` | 0，全参数训练 |
| `VAL_FREQ` | 5 |
| `SAVE_FREQ` | 5 |

每个 step 使用 32 个问题并生成 128 条回答。`GLOBAL_BATCH_SIZE=16`，因此每次 rollout 分成 2 个 Actor minibatch。训练期间单卡最高分配显存约 13.4GB，最高保留显存约 18.8GB，没有发生 CUDA OOM，说明 4 张 RTX 3090 能够承载这组参数。

## 2. 完整训练过程

Geometry3K 过滤过长样本后，每个 epoch 为 65 step，因此原计划的 15 epoch 约为 975 step：

```text
65 step/epoch × 15 epoch = 975 step
```

首次训练运行到 Step 160，约为 2.46 epoch。此时多个 Ray worker 进入不可中断的 `D` 状态，`nvidia-smi` 无法返回，系统日志持续出现：

```text
NVRM: Xid 74, NVLink MINION fatal error
```

训练日志中没有 OOM、NCCL timeout 或 Python traceback，Step 160 的验证和 checkpoint 也已完成，因此中断原因是集群 GPU/NVLink 的底层故障，而不是训练参数或代码错误。管理员同时说明该集群开机运行一段时间后存在 GPU 掉卡问题。

考虑到继续运行 975 step 很可能再次中断，本次实验将目标缩短为 3 epoch，即 Step 195：

```text
65 step/epoch × 3 epoch = 195 step
```

节点恢复后，从 `global_step_160` 继续训练。checkpoint 包含模型、优化器、额外训练状态和 DataLoader 状态，因此能够恢复完整训练进度。续训保持原始 batch size、学习率、KL 系数和并行配置不变，只增加 checkpoint 路径，并使用 `trainer.max_steps=195` 限定停止位置：

```bash
export WANDB_ENTITY="kisaragi-ustc"
export WANDB_RUN_ID="qn375g4y"
export WANDB_RESUME="must"

RUN_DIR="/data/jinda/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1/checkpoints/Qwen2.5-VL-3B GRPO Reproduction with EasyR1/qwen2_5_vl_3b_geo_grpo_3090_full"
CKPT="$RUN_DIR/global_step_160"
MODEL_DIR="/data/jinda/models/Qwen2.5-VL-3B-Instruct"
WANDB_PROJECT_NAME="Qwen2.5-VL-3B GRPO Reproduction with EasyR1"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH="$MODEL_DIR" \
N_GPUS=4 \
ROLLOUT_BATCH_SIZE=32 \
GLOBAL_BATCH_SIZE=16 \
MICRO_BATCH_SIZE_UPDATE=1 \
MICRO_BATCH_SIZE_EXPERIENCE=1 \
ROLLOUT_N=4 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=512 \
MAX_PIXELS=524288 \
GPU_MEMORY_UTILIZATION=0.5 \
LEARNING_RATE=1e-6 \
KL_COEF=1e-2 \
LORA_RANK=0 \
TOTAL_EPOCHS=15 \
VAL_FREQ=5 \
SAVE_FREQ=5 \
EXPERIMENT_NAME=qwen2_5_vl_3b_geo_grpo_3090_full \
LOGGER="['file','wandb']" \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh \
trainer.project_name="'$WANDB_PROJECT_NAME'" \
trainer.load_checkpoint_path="$CKPT" \
trainer.max_steps=195
```

`TOTAL_EPOCHS=15` 用于保持原始配置，实际停止位置由优先级更高的 `trainer.max_steps=195` 控制。续训继续写入原 W&B run，并最终正常完成 Step 195。最终 checkpoint 大小约 29GB，说明模型和训练状态均已成功保存。

## 3. 训练结果

### 3.1 训练前后对比

训练开始前执行了一次完整验证。下表将训练前基线、Step 160 和最终 Step 195 放在一起比较：

| 验证指标 | 训练前 Step 0 | Step 160 | Step 195 |
| --- | ---: | ---: | ---: |
| Overall reward | 0.0208 | 0.6589 | 0.6506 |
| Accuracy reward | 0.0383 | 0.3328 | 0.3245 |
| Format reward | 0.0033 | 0.9850 | 0.9767 |
| 平均 response length | 368.98 | 220.58 | 239.88 |
| Response clip ratio | 0.2772 | 0.0164 | 0.0235 |

最终验证 overall reward 比训练前提高 0.6298，accuracy reward 从 0.0383 提高到 0.3245，format reward 则从接近 0 提高到 0.9767。训练首先快速解决了输出格式问题，同时也提高了答案正确率。响应平均长度明显下降，撞到 512 token 上限的比例从 27.72% 降至 2.35%，说明奖励提升不是依靠持续生成更长回答获得的。

三个 epoch 结束位置的训练 batch 指标如下：

| 训练指标 | Step 65 | Step 130 | Step 195 |
| --- | ---: | ---: | ---: |
| Overall reward | 0.5430 | 0.6289 | 0.5859 |
| Accuracy reward | 0.1328 | 0.2656 | 0.1875 |
| Format reward | 0.9531 | 0.9922 | 0.9844 |

单个训练 batch 的方差较大，不能把 Step 195 低于 Step 130 直接解释为能力退化。例如末段 Step 188 和 Step 191 的 accuracy reward 分别达到 0.3906 和 0.4219，而 Step 195 为 0.1875。因此最终能力判断应以完整验证集为主。

### 3.2 奖励曲线

验证 overall reward 在 Step 0 为 0.0208，Step 5 已提高到 0.2488，Step 25 达到 0.5549，之后增速明显放缓。Step 80 和 Step 120 分别达到 0.6398 和 0.6597，完整训练中的最高验证值出现在 Step 180，为 0.6639。最终 Step 195 为 0.6506，比峰值低 0.0133。

这条曲线可以分成两个阶段：

1. 前约 25 step 主要学习输出格式，format reward 很快接近 1。
2. 后续提升主要来自 accuracy reward，但增长速度较慢，并伴随明显 batch 波动。

Step 160 至 Step 195 的验证 overall reward 从 0.6589 变为 0.6506，accuracy reward 从 0.3328 变为 0.3245。期间虽出现 Step 180 的局部峰值，但没有形成持续上升趋势，说明训练在第三个 epoch 进入阶段性平台。验证指标没有持续下跌，因此现有数据也不足以认定发生了明显过拟合。

格式奖励在训练中后期长期保持在约 0.96 至 1.00。此时 overall reward 近似由格式奖励和准确率奖励共同决定，继续优化的主要瓶颈已经从格式遵循转为几何题求解。

### 3.3 优化稳定性

| 指标 | Step 160 | Step 195 |
| --- | ---: | ---: |
| PPO KL | 0.000204 | 0.000206 |
| KL loss | 0.0632 | 0.0507 |
| Grad norm | 0.7891 | 0.5449 |
| PPO higher clip fraction | 0.0000369 | 0.0001921 |
| PPO lower clip fraction | 0 | 0 |

PPO KL 始终处于很低的量级，KL loss 虽有波动，但没有持续失控。绝大多数grad norm 低于配置中的 `max_grad_norm=1.0`；少数 step 出现超过 1 的尖峰，例如 Step 71 为 1.6406，但没有演变为连续增长。PPO clip fraction 也长期接近0，说明更新很少触发裁剪。结合 loss、KL 和梯度指标，训练没有出现数值发散迹象。

### 3.4 响应长度

训练前验证回答平均长度为 368.98 token，Step 160 降到 220.58，Step 195 为 239.88。训练 batch 的平均长度在后期大多位于约 200 至 270 token，最终 Step 195 为 270.09。验证与训练 batch 的统计口径不同，因此不应直接用两个单点判断长度回升，但两者的 clip ratio 都很低。

总体上，模型在学会格式后减少了大量达到长度上限的回答，没有出现通过冗长输出投机获取格式奖励的明显迹象。

### 3.5 运行效率

普通训练 step 通常耗时约 115 至 123 秒。以 Step 195 为例，主要耗时如下：

| 阶段 | 耗时 |
| --- | ---: |
| Rollout 生成 | 25.09 秒 |
| 旧策略 log-prob | 17.97 秒 |
| 参考策略 log-prob | 17.23 秒 |
| Actor 更新 | 61.82 秒 |
| 验证 | 258.40 秒 |
| 保存 checkpoint | 24.17 秒 |
| Step 总耗时 | 404.73 秒 |

普通 step 中 Actor 更新是最大的单项开销。每 5 step 执行验证和保存时，总耗时会增加到约 375 至 405 秒；Step 195 的验证约占总耗时 63.8%。`VAL_FREQ=5` 提供了较密集的曲线，但也是训练总时间的重要组成部分。

### 3.6 生成样例限制

本次整理收到的是逐 step 指标，没有包含 Step 0 与 Step 195 的具体 `prompt`、`output`、`ground_truth` 和 `score`。因此可以确认格式通过率、答案正确率和长度分布的变化，但不能仅凭聚合奖励可靠判断推理过程是否更严谨、错误类型是否改变。生成样例的定性比较需要另行保留 `generations.log` 后完成，本文不据此虚构案例。

## 4. 结论
本次实验原计划训练 15 epoch，但受 RTX 3090 集群长时间运行后 GPU 掉卡的影响，最终在第 3 个 epoch，也就是 Step 195 结束。断点续训成功恢复了模型、优化器、数据进度和原 W&B run，证明 EasyR1 的 checkpoint 恢复流程能够正常工作。

3 epoch 训练已经让模型稳定遵循指定输出格式，最终验证 accuracy reward 为 0.3245，明显高于训练前的 0.0383，说明 GRPO 训练产生了可观察的效果。不过，验证 overall reward 在 Step 180 达到 0.6639 后，Step 195 为 0.6506；Step 160 到 Step 195 的验证指标整体处于同一区间，尚不能证明模型已经充分收敛，也不能据此判断继续训练一定会提高准确率。

因此，本次结果更适合作为 Qwen2.5-VL-3B 全参数 GRPO 训练流程、硬件资源需求和断点恢复能力的验证，而不是模型最终能力上限的评估。
