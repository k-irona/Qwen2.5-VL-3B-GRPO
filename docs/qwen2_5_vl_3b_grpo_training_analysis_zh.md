# Qwen2.5-VL-3B GRPO 训练分析

本文记录 Qwen2.5-VL-3B-Instruct 在 Geometry3K 上进行 GRPO 训练的实际过程，
重点分析训练参数、训练中断与恢复、训练轮数调整，以及当前阶段的指标表现。
环境安装过程参见
[《使用 Conda 配置 EasyR1 训练环境》](easyr1_conda_environment_zh.md)，
本文不再重复。

## 1. 实验设置

实验使用 EasyR1，在 4 张 RTX 3090 24GB 上进行 BF16 全参数训练。数据集为
Geometry3K，使用 R1V 提示模板和规则奖励函数。奖励由两部分组成：

- `format reward`：回答是否符合指定格式。
- `accuracy reward`：最终答案是否正确。

GRPO 对同一个问题生成多条回答，并在组内计算相对优势。本次实验每个问题
生成 4 条候选回答，不额外训练 critic 模型。

### 1.1 主要训练参数

| 参数 | 实际值 | 说明 |
| --- | ---: | --- |
| GPU 数量 | 4 | 使用 GPU 0-3 |
| `ROLLOUT_BATCH_SIZE` | 32 | 每个训练 step 使用 32 个问题 |
| `ROLLOUT_N` | 4 | 每个问题生成 4 条回答 |
| `GLOBAL_BATCH_SIZE` | 16 | Actor 更新 minibatch 大小 |
| `MICRO_BATCH_SIZE_UPDATE` | 1 | 每张 GPU 的更新 micro batch |
| `MICRO_BATCH_SIZE_EXPERIENCE` | 1 | 每张 GPU 的 log-prob micro batch |
| `MAX_PROMPT_LENGTH` | 2048 | 最大输入 token 数 |
| `MAX_RESPONSE_LENGTH` | 512 | 最大生成 token 数 |
| `MAX_PIXELS` | 524288 | 单张图片最大像素数 |
| `GPU_MEMORY_UTILIZATION` | 0.5 | vLLM 显存使用比例 |
| `LEARNING_RATE` | `1e-6` | Actor 学习率 |
| `KL_COEF` | `1e-2` | KL 正则系数 |
| `LORA_RANK` | 0 | 关闭 LoRA，进行全参数训练 |
| `VAL_FREQ` | 5 | 每 5 step 验证一次 |
| `SAVE_FREQ` | 5 | 每 5 step 保存一次 |

每个 step 实际生成的回答数为：

```text
32 个问题 × 4 条回答 = 128 条回答
```

由于 `ROLLOUT_BATCH_SIZE=32`、`GLOBAL_BATCH_SIZE=16`，每个 rollout 会被划分为
2 个 Actor minibatch，这与日志中的 `Train mini-batches: 2/2` 一致。

训练期间单卡最高分配显存约为 13.4GB，最高保留显存约为 18.8GB，没有出现
CUDA OOM。因此，RTX 3090 的显存能够支持这组参数，后续中断并不是模型或
batch size 超出显存能力造成的。

## 2. Step 和 Epoch 的计算

Geometry3K 过滤过长样本后，训练 DataLoader 每个 epoch 约有 65 step。
在没有设置 `mini_rollout_batch_size` 时，训练步数可以理解为：

```text
每个 epoch 的 step ≈ 训练样本数 / ROLLOUT_BATCH_SIZE
总 step = 每个 epoch 的 step × epoch 数
```

最初计划训练 15 epoch：

```text
65 step/epoch × 15 epoch = 975 step
```

因此，`step 160` 约对应：

```text
160 / 65 ≈ 2.46 epoch
```

第三个 epoch 的结束位置约为：

```text
65 step/epoch × 3 epoch = 195 step
```

## 3. 为什么没有完成 15 Epoch

训练最初按 15 epoch 启动，但运行到 `step 160` 后不再前进。四个 Ray worker
均进入不可中断的 `D` 状态，等待位置为：

```text
rwsem_down_write_slowpath
```

与此同时，`nvidia-smi` 也无法正常返回。系统日志最终记录了 PCI 地址
`0000:ce:00.0` 对应 GPU 的连续错误：

```text
NVRM: Xid 74, NVLink MINION fatal error
```

这类错误发生在 NVIDIA 驱动、NVLink 或硬件通信层。训练日志中没有 OOM、
NCCL timeout 或 Python traceback；Step 160 的验证和 checkpoint 保存也已经
完成。因此，这次停止不是训练代码或参数配置错误，而是集群 GPU 掉卡导致的
底层故障。

管理员说明该 RTX 3090 集群在开机运行一段时间后存在 GPU 掉卡问题。继续执行
完整的 975 step，意味着训练很可能在后续再次因节点故障中断。综合训练成本和
机器稳定性，本次实验将目标从 15 epoch 调整为 3 epoch，即训练到 Step 195。

这个调整属于基础设施限制下的实验取舍。3 epoch 的结果可以用于确认 GRPO
训练是否产生了效果，但不能视为完整收敛结果，也不能与 15 epoch 基线作严格
等价比较。

## 4. Step 160 的阶段性结果

故障发生时，最新完整日志为 Step 160，主要指标如下：

| 指标 | Step 160 |
| --- | ---: |
| 训练 overall reward | 0.6172 |
| 训练 accuracy reward | 0.2422 |
| 训练 format reward | 0.9922 |
| 验证 overall reward | 0.6589 |
| 验证 accuracy reward | 0.3328 |
| 验证 format reward | 0.9850 |
| PPO KL | 0.000204 |
| KL loss | 0.0632 |
| Grad norm | 0.7891 |
| 平均 response length | 219.68 |
| Response clip ratio | 0.0078 |

### 4.1 格式学习

训练和验证的 format reward 分别达到 0.9922 和 0.9850，说明模型已经基本
掌握了要求的输出格式。格式奖励接近饱和后，overall reward 的进一步提高将
更多依赖答案正确率，而不是单纯改善回答结构。

### 4.2 准确率表现

验证 accuracy reward 达到 0.3328，已经能够从 reward 曲线中观察到一定的
训练效果。不过，训练在约 2.46 epoch 时中断，准确率仍有继续上升的空间。
因此，这一结果说明 GRPO 更新方向有效，但还不能证明模型已经收敛。

### 4.3 训练稳定性

Step 160 的 PPO KL 约为 0.000204，数值较低；grad norm 约为 0.789，也未
达到 `max_grad_norm=1.0` 的裁剪上限。Response clip ratio 只有约 0.78%，
说明绝大多数回答没有撞到 512 token 的长度限制。

从这些指标看，在 GPU 故障发生前，优化过程本身没有明显发散迹象。训练停止
与模型数值稳定性无关。

### 4.4 训练耗时

Step 160 总耗时约为 386 秒，其中：

| 阶段 | 耗时 |
| --- | ---: |
| 生成回答 | 22.91 秒 |
| 旧策略 log-prob | 16.63 秒 |
| 参考策略 log-prob | 15.84 秒 |
| Actor 更新 | 61.43 秒 |
| 验证 | 244.36 秒 |
| 保存 checkpoint | 25.09 秒 |

该 step 同时执行了验证和 checkpoint 保存，因此耗时明显高于普通训练 step。
其中验证约占 63%，说明 `VAL_FREQ=5` 提供了密集的评估数据，但也显著增加了
总训练时间。本次实验为了保持续训前后的配置一致，没有在中途修改验证频率。

## 5. 从 Step 160 恢复训练

Step 160 的 checkpoint 已经完整保存，包含：

- 4 个模型分片。
- 4 个优化器分片。
- 4 个额外状态分片。
- DataLoader 状态。

因此可以恢复模型、优化器、随机状态和数据读取位置，而不需要从 Step 0
重新训练。

### 5.1 恢复前准备

节点重启并确认 `nvidia-smi` 正常后，设置原模型和 checkpoint 路径：

```bash
RUN_DIR="/data/jinda/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1/checkpoints/Qwen2.5-VL-3B GRPO Reproduction with EasyR1/qwen2_5_vl_3b_geo_grpo_3090_full"
CKPT="$RUN_DIR/global_step_160"
MODEL_DIR="/data/jinda/models/Qwen2.5-VL-3B-Instruct"
WANDB_PROJECT_NAME="Qwen2.5-VL-3B GRPO Reproduction with EasyR1"
```

为了继续写入原来的 WandB run，需要复用原 run ID：

```bash
export WANDB_ENTITY="kisaragi-ustc"
export WANDB_RUN_ID="qn375g4y"
export WANDB_RESUME="must"
```

### 5.2 恢复命令

续训时保持 batch size、学习率、KL、并行方式和随机种子等训练参数不变，只
增加 checkpoint 路径，并通过 `trainer.max_steps=195` 在第三个 epoch 结束：

```bash
unset LD_LIBRARY_PATH

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

这里保留 `TOTAL_EPOCHS=15` 是为了维持原始配置。`trainer.max_steps=195`
优先级更高，仅用于限定本次实验的实际停止位置。正常恢复时日志应出现：

```text
Load from checkpoint: .../global_step_160
Total training steps: 195
Running step: 160/195
```

如果没有出现 checkpoint 加载信息，或者训练从 Step 0 开始，应立即停止并
检查路径，避免覆盖原实验日志。

## 6. 结果总结

本次实验受集群稳定性限制，从原计划的 15 epoch 缩短为 3 epoch。当前已经
确认的 Step 160 结果表明：

1. 四张 RTX 3090 能够在现有参数下完成 Qwen2.5-VL-3B 的全参数 GRPO 训练，
   显存不是主要瓶颈。
2. 模型已经稳定学会指定输出格式，format reward 接近 1。
3. 验证 accuracy reward 达到约 0.333，reward 曲线已经表现出早期训练效果。
4. PPO KL、grad norm 和输出截断比例均无明显异常，训练没有数值发散迹象。
5. 中断原因是集群的 GPU/NVLink `Xid 74` 故障，而不是训练参数或代码错误。

最终实验将在 Step 195，也就是第三个 epoch 结束时停止。由于训练轮数较短，
本文结论应理解为对训练流程和早期效果的验证，而不是对模型最终能力上限的
评估。获得 Step 195 指标后，可以在本节补充最终验证准确率，并与 Step 0 和
Step 160 进行完整对照。
