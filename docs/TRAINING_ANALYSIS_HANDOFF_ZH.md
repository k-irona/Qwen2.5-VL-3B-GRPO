# Qwen2.5-VL-3B GRPO 训练分析交接文档

本文档用于在新对话中继续分析训练结果，并将
`docs/qwen2_5_vl_3b_grpo_training_analysis_zh.md` 完善为可发布的最终版本。

## 1. 当前任务

训练原计划运行 15 个 epoch，后来因 Sui 服务器长时间运行后出现 GPU/NVLink
故障，调整为训练到第 3 个 epoch，即 Step 195。

当前已完成的工作：

- 环境配置文档已经整理完成。
- 训练分析正文已经写出主体结构。
- 已记录初始训练参数、Step 160 指标、硬件故障和断点续训方案。
- 已从完整的 Step 160 checkpoint 准备继续训练。

仍待完成的工作：

- 确认续训是否实际到达 Step 195。
- 收集 Step 195 的训练、验证和性能指标。
- 对比训练初期、Step 160 和 Step 195 的曲线。
- 更新训练分析正文中的“待补充”内容和最终结论。

## 2. 需要先阅读的文件

新对话开始后应先阅读：

1. `docs/easyr1_conda_environment_zh.md`
2. `docs/qwen2_5_vl_3b_grpo_training_analysis_zh.md`
3. 本交接文档
4. `examples/qwen2_5_vl_3b_geo3k_grpo.sh`
5. `examples/config.yaml`

博客标题分别为：

- `使用 Conda 配置 EasyR1 训练环境`
- `Qwen2.5-VL-3B GRPO 训练分析`

环境文档只负责环境搭建。训练分析文档负责参数、训练过程、故障、断点续训和结果分析，
不要在两篇文章中大段重复安装步骤。

## 3. 已确认的训练配置

| 配置项 | 实际值 |
| --- | ---: |
| 模型 | Qwen2.5-VL-3B-Instruct |
| 数据集 | hiyouga/geometry3k |
| 算法 | GRPO |
| GPU | 4 张 RTX 3090，设备 0、1、2、3 |
| rollout batch size | 32 |
| global batch size | 16 |
| update micro batch size | 1/GPU |
| experience micro batch size | 1/GPU |
| 每个问题的 rollout 数量 | 4 |
| 最大 prompt 长度 | 2048 |
| 最大 response 长度 | 512 |
| 最大图像像素 | 524288 |
| 学习率 | 1e-6 |
| KL 系数 | 1e-2 |
| LoRA rank | 0，全参数训练 |
| vLLM GPU memory utilization | 0.5 |
| vLLM tensor parallel size | 4 |
| actor 精度 | BF16 |
| PPO epochs | 1 |
| validation frequency | 5 steps |
| checkpoint frequency | 5 steps |
| 原计划 epoch | 15 |
| 最终计划 epoch | 3 |
| 最终目标 step | 195 |

训练参数在续训时必须与原始训练保持一致。续训只额外设置 checkpoint 路径、
W&B 恢复变量和 `trainer.max_steps=195`。

## 4. Step 与 Epoch 的对应关系

Geometry3K 训练集约有 3000 个样本，`rollout_batch_size=32`。过滤超长输入后，
本次实际运行每个 epoch 为 65 个训练 step。

因此：

- Epoch 1 结束：Step 65
- Epoch 2 结束：Step 130
- Step 160：约 2.46 epoch
- Epoch 3 结束：Step 195
- 原计划 15 epoch：约 Step 975

`global_batch_size=16` 控制 actor 参数更新时的全局 batch，不直接决定每个 epoch
包含多少 step。每个 epoch 的 step 数主要由有效训练样本数和
`rollout_batch_size=32` 决定。

## 5. Step 160 已确认结果

Step 160 的 checkpoint 已完整保存。已有指标如下：

| 指标 | Step 160 |
| --- | ---: |
| train/reward/overall | 0.6172 |
| train/reward/accuracy | 0.2422 |
| train/reward/format | 0.9922 |
| val/reward/overall | 0.6589 |
| val/reward/accuracy | 0.3328 |
| val/reward/format | 0.9850 |
| PPO KL | 0.000204 |
| KL loss | 0.0632 |
| gradient norm | 0.7891 |
| 平均响应长度 | 219.68 |
| clip ratio | 0.0078 |
| 单步总耗时 | 约 386 秒 |
| 验证耗时 | 约 244.36 秒 |

已有的谨慎结论：

- 格式奖励接近 1，模型已经稳定遵守输出格式。
- 准确率奖励仍明显低于格式奖励，数学求解能力仍是主要瓶颈。
- 验证总体奖励和准确率高于该步训练 batch，但单步 batch 波动较大，不能仅凭一个点
  判断泛化能力。
- KL、梯度范数和裁剪比例未显示明显发散。
- Step 160 尚不能称为收敛，但 reward 曲线已经能看到训练效果。

## 6. 中断原因

训练中断不是显存不足，也没有证据表明是普通 Python、Ray 或 EasyR1 代码异常。

已观察到：

- 多个 Ray worker 进入不可中断的 `D` 状态。
- `nvidia-smi` 无法正常返回。
- 内核日志持续出现 PCI 地址 `0000:ce:00` 的 NVIDIA Xid 74。
- 日志明确显示 `NVLink MINION` fatal error。
- Link 0 至 Link 3 被连续禁用。
- 管理员说明 Sui 机器开机运行一段时间后会发生 GPU 掉卡。

因此，训练由底层 GPU/NVLink/驱动或硬件状态异常终止。Ray 日志为空或主进程只收到
SIGTERM，是底层故障之后的表象，不是根因。

由于长时间连续运行的可靠性不足，实验目标从 15 epoch 缩短为 3 epoch。最终文章应将
这一点表述为实验基础设施限制，而不是算法主动早停或模型已经完全收敛。

## 7. Step 160 Checkpoint

checkpoint 根目录：

```text
/data/jinda/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1/checkpoints/Qwen2.5-VL-3B GRPO Reproduction with EasyR1/qwen2_5_vl_3b_geo_grpo_3090_full
```

续训 checkpoint：

```text
global_step_160
```

该 checkpoint 已确认包含：

- 4 个 actor model shard
- 4 个 optimizer shard
- 4 个 extra state shard
- DataLoader 状态

因此能够恢复模型、优化器、训练步数和数据读取进度，而不是只加载模型权重重新训练。

## 8. W&B 信息

- Entity：`kisaragi-ustc`
- Project：`Qwen2.5-VL-3B GRPO Reproduction with EasyR1`
- Run ID：`qn375g4y`
- 恢复策略：`WANDB_RESUME=must`

Run URL：

```text
https://wandb.ai/kisaragi-ustc/Qwen2.5-VL-3B%20GRPO%20Reproduction%20with%20EasyR1/runs/qn375g4y
```

恢复到同一个 W&B run 的关键变量：

```bash
export WANDB_ENTITY=kisaragi-ustc
export WANDB_RUN_ID=qn375g4y
export WANDB_RESUME=must
```

最终分析应优先使用 W&B 的完整曲线。项目的本地 FileLogger 在新进程启动时可能以写入
模式重新创建 `experiment_log.jsonl`，因此续训后的本地日志可能只包含后半段。

同时检查是否存在以下续训前备份：

```text
experiment_log_before_resume_step160.jsonl
generations_before_resume_step160.log
```

## 9. 首次续训失败及处理

第一次执行续训命令时尚未加载 checkpoint，程序在创建 DataLoader 时失败：

```text
Couldn't reach 'hiyouga/geometry3k' on the Hub (LocalEntryNotFoundError)
```

直接原因是 Hugging Face 连接被重置，并且当前缓存中没有可直接使用的数据集副本。

当时还发现：

- 根分区 `/` 仅剩约 7 GB，使用率显示为 100%。
- `/data` 仍有约 4.9 TB 可用空间。
- `/tmp/ray` 本身只有约 2.1 MB，不是主要占用来源。

后续运行应将临时目录和缓存放到 `/data/jinda`：

```bash
export TMPDIR=/data/jinda/tmp
export RAY_TMPDIR=/data/jinda/tmp/ray
export HF_HOME=/data/jinda/cache/huggingface
export TORCH_HOME=/data/jinda/cache/torch
export TRITON_CACHE_DIR=/data/jinda/cache/triton
export HF_ENDPOINT=https://hf-mirror.com
```

## 10. 最终结果出来后需要收集的材料

在服务器上进入项目目录后，先定义：

```bash
RUN_DIR="/data/jinda/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1/checkpoints/Qwen2.5-VL-3B GRPO Reproduction with EasyR1/qwen2_5_vl_3b_geo_grpo_3090_full"
```

检查最终 checkpoint：

```bash
find "$RUN_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V
find "$RUN_DIR/global_step_195/actor" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
du -sh "$RUN_DIR/global_step_195"
```

检查本地日志及备份：

```bash
ls -lh "$RUN_DIR"/experiment_log*.jsonl "$RUN_DIR"/generations*.log 2>/dev/null
tail -n 3 "$RUN_DIR/experiment_log.jsonl"
```

检查训练结束前后的系统错误：

```bash
dmesg -T | grep -E 'NVRM|Xid|NVLink' | tail -n 100
nvidia-smi
nvidia-smi nvlink --status
```

还需要保留：

- 训练程序最后约 200 行输出。
- W&B 中 Step 160 至 Step 195 的曲线截图或导出数据。
- Step 195 的 validation 指标。
- Step 195 附近的 generation 示例。
- 是否正常退出，以及是否成功保存 `global_step_195`。

## 11. 最终分析重点

拿到结果后，至少完成以下对比：

| 维度 | 需要回答的问题 |
| --- | --- |
| 总体奖励 | Step 195 是否继续高于训练初期和 Step 160？ |
| 准确率奖励 | 数学求解能力是否继续改善，还是进入平台期？ |
| 格式奖励 | 是否持续稳定在接近 1 的水平？ |
| 训练与验证 | 两条曲线是否同向，是否出现明显过拟合迹象？ |
| KL | 策略与参考模型的偏移是否受控？ |
| 梯度范数 | 是否稳定，是否出现尖峰或异常值？ |
| Clip ratio | PPO 更新是否频繁触发裁剪？ |
| 响应长度 | 奖励变化是否伴随异常变长或变短？ |
| 样例质量 | 推理格式、计算过程和最终答案有什么实际变化？ |
| 运行效率 | rollout、更新、验证和保存分别消耗多少时间？ |

不要只比较 Step 160 和 Step 195 两个离散点。应结合 W&B 曲线观察滑动趋势和 batch
波动，并明确只有 3 epoch，结论属于阶段性实验结果。

## 12. 对训练分析正文的修改要求

最终应更新：

```text
docs/qwen2_5_vl_3b_grpo_training_analysis_zh.md
```

需要完成：

1. 将“Step 195 待补充”替换为实际结果。
2. 增加训练初期、Step 160、Step 195 的对比表。
3. 描述 reward、accuracy、format、KL 和梯度曲线的整体趋势。
4. 加入代表性生成样例的定性分析。
5. 说明续训是否成功恢复 optimizer、step 和 W&B run。
6. 说明 Step 195 是否正常保存。
7. 将最终结论限制在 3 epoch 的证据范围内。

如果 Step 195 没有完成，不要伪造结果。应记录实际停止 step、最后完整 checkpoint 和
新的故障信息，再按实际进度修改标题或结论。

## 13. 建议的新对话开场

可以在新对话中直接发送：

```text
请先阅读 docs/TRAINING_ANALYSIS_HANDOFF_ZH.md 和其中列出的文档。
训练已经结束，下面是最终日志、W&B 数据和 checkpoint 信息。
请据此分析完整训练过程，并把
docs/qwen2_5_vl_3b_grpo_training_analysis_zh.md 完善成可发布的最终版本。
不要改变已经确认的训练参数，也不要把 3 epoch 的结果表述为充分收敛。
```

随后附上最终日志、指标、曲线截图或导出的 CSV。
