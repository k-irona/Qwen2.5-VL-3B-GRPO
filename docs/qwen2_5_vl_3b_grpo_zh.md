# Qwen2.5-VL-3B EasyR1 GRPO 训练与可视化

## 1. 推荐起点

仓库中的启动脚本已经配置 Geometry3K、R1V 提示模板和奖励函数：

```bash
wandb login
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh
```

当前按 4 张 RTX 3090 24GB 的全参数 BF16 配置运行。日志同时写入 WandB 和：

```text
checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16/
```

首次排查流程时，建议先运行 10 步：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 \
TOTAL_EPOCHS=1 VAL_FREQ=5 SAVE_FREQ=5 \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh trainer.max_steps=10
```

脚本会把额外参数原样传给 Python，因此也可以直接追加 OmegaConf 覆盖项。

## 2. 核心参数

| 参数 | 默认值 | 调整建议 |
| --- | ---: | --- |
| `ROLLOUT_BATCH_SIZE` | 32 | 首要吞吐参数，必须能被 `GLOBAL_BATCH_SIZE` 整除 |
| `GLOBAL_BATCH_SIZE` | 16 | 一次 rollout 上做 actor 更新的 minibatch 大小 |
| `ROLLOUT_N` | 4 | 每个问题采样数；GRPO 必须大于 1，通常用 4-8 |
| `MAX_RESPONSE_LENGTH` | 512 | 推理输出不足时增大，OOM 或过慢时减小 |
| `MAX_PIXELS` | 524288 | 图像 token 的主要来源，图形细节不足时逐步增大 |
| `LORA_RANK` | 0 | 0 表示全参数训练；显存不足时可设为 32 切换到 LoRA |
| `LORA_ALPHA` | 64 | 仅在启用 LoRA 时生效 |
| `GPU_MEMORY_UTILIZATION` | 0.5 | vLLM 可使用的显存比例，稳定后可尝试 0.55-0.6 |
| `LEARNING_RATE` | `1e-6` | 全参数训练基线；KL 过快上升时降到 `5e-7` |
| `KL_COEF` | `1e-2` | 输出漂移过快时增大，学习停滞时减小 |
| `VAL_FREQ` | 5 | 每隔多少 step 做固定验证 |

显存不足时按这个顺序调整：

1. `MAX_PIXELS=393216`
2. `MAX_RESPONSE_LENGTH=384`
3. `GPU_MEMORY_UTILIZATION=0.45`
4. 保持 `MICRO_BATCH_SIZE_UPDATE=1` 和 `MICRO_BATCH_SIZE_EXPERIENCE=1`
5. `ROLLOUT_BATCH_SIZE=16 GLOBAL_BATCH_SIZE=8`
6. 仍然 OOM 时使用 `LORA_RANK=32 LEARNING_RATE=1e-5`

脚本使用 `worker.actor.fsdp.torch_dtype=bf16` 和
`worker.actor.optim.strategy=adamw_bf16` 开启 BF16 全参数训练。仓库给出的
3B BF16 资源估算是 1 张 40GB GPU，而 AMP 需要 4 张 40GB GPU。RTX 3090
属于 Ampere 架构并支持 BF16。当前 4 卡实测每卡显存峰值约为 13.5GB
allocated、18.9GB reserved，但实际峰值仍受图像尺寸和生成长度影响。

## 3. 训练前后对比

配置中的 `trainer.val_before_train=true` 会在 step 0 测基础模型。训练结束时 EasyR1 会再次验证，所以同一次实验内可直接比较：

- `val/reward_score`
- `val/accuracy_reward`
- `val/format_reward`
- `val/generations`

WandB 中打开 run，查看上述曲线和 `val/generations` 表格即可。需要纯本地报告时执行：

```bash
python3 scripts/visualize_training.py \
  checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16
```

输出文件：

```text
checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16/training_report.html
```

该报告包含训练曲线，以及 step 0 和最终 step 在相同验证问题上的回答与得分对照。

## 4. 判断训练是否正常

- `val/accuracy_reward` 上升才代表任务正确率提升，不能只看总 reward。
- `val/format_reward` 很快到 1、accuracy 不涨，说明模型只学会了格式。
- `actor/ppo_kl` 持续快速上升时，提高 `KL_COEF` 或降低学习率。
- `response_length/clip_ratio` 偏高时，答案经常撞到长度上限，应提高 `MAX_RESPONSE_LENGTH`。
- 训练 reward 上升而验证 reward 下降，通常是奖励投机或过拟合，应检查生成样例并加强奖励函数。

Geometry3K 的仓库基线是 Qwen2.5-VL-3B 测试准确率约 `0.24 -> 0.38`，使用全参数 AMP、学习率 `1e-6`、KL `1e-2`。不同硬件、版本和随机种子会有差异。

## 5. 等待 GPU 空闲后自动训练

等待任意四张 GPU 连续 3 分钟空闲后启动：

```bash
NUM_GPUS=4 \
nohup bash scripts/wait_for_gpus_and_train.sh \
  > wait_for_gpus.log 2>&1 &
```

只允许从 GPU 4、5、6、7 中选择四张：

```bash
GPU_CANDIDATES=4,5,6,7 \
NUM_GPUS=4 \
nohup bash scripts/wait_for_gpus_and_train.sh \
  > wait_for_gpus.log 2>&1 &
```

查看等待和训练日志：

```bash
tail -f wait_for_gpus.log
```

默认空闲条件是显存占用小于 2000 MiB、利用率小于 10%，每 60 秒检查一次，
连续满足 3 次才启动。可以通过 `MAX_MEMORY_USED_MB`、
`MAX_GPU_UTILIZATION`、`POLL_INTERVAL_SECONDS` 和
`REQUIRED_FREE_CHECKS` 覆盖。
