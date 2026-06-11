# Qwen2.5-VL-3B GRPO Reproduction with EasyR1

[English](README.md) | [简体中文](README_zh.md)

This repository reproduces the GRPO training workflow for
[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
on the [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) dataset.
It is built on [EasyR1](https://github.com/hiyouga/EasyR1), with a training
configuration validated on four NVIDIA GeForce RTX 3090 24 GB GPUs selected
from an eight-GPU server.

The default experiment uses full-parameter BF16 training, vLLM rollout,
FSDP, rule-based rewards, Weights & Biases tracking, local logging, automatic
GPU waiting, and before/after generation comparison.

## Training Overview

For each prompt, the current policy generates four candidate responses.
The reward function evaluates answer accuracy and output format. GRPO
normalizes rewards within each response group and updates the policy without
training an additional critic model. A KL penalty limits excessive deviation
from the reference policy.

The expected response format is:

```text
<think>reasoning process</think><answer>final answer</answer>
```

Main components:

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Dataset: `hiyouga/geometry3k`
- Algorithm: GRPO
- Precision: BF16
- Training mode: full-parameter fine-tuning
- Distributed strategy: FSDP on four GPUs
- Rollout engine: vLLM
- Reward: answer accuracy and response format
- Tracking: local JSONL logs and Weights & Biases

## Requirements

- Linux
- Python 3.9+
- NVIDIA GPUs with BF16 support
- CUDA environment compatible with PyTorch, FlashAttention, and vLLM
- Server: 8 x NVIDIA GeForce RTX 3090 24 GB
- Training devices: GPU 0-3 (`CUDA_VISIBLE_DEVICES=0,1,2,3`)
- NVIDIA driver: 530.30.02 (`nvidia-smi` reports CUDA 12.1)
- Conda environment: PyTorch 2.8.0 cu126 with CUDA Toolkit 12.6

The upstream EasyR1 hardware table estimates that Qwen2.5-VL-3B full
fine-tuning requires `1 x 40 GB` in BF16. Actual memory usage depends on image
resolution, prompt length, response length, and rollout settings.

For the verified RTX 3090 environment based on Python 3.10, PyTorch 2.8.0 cu126,
vLLM 0.11.0, and FlashAttention 2.8.3, see
[docs/easyr1_conda_environment_zh.md](docs/easyr1_conda_environment_zh.md).

## Installation

Using the prebuilt EasyR1 image is recommended:

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all \
  -v "$PWD":/workspace/Qwen2.5-VL-3B-GRPO \
  hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

Alternatively, install the project in an existing compatible environment:

```bash
git clone https://github.com/k-irona/Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1.git
cd Qwen2.5-VL-3B-GRPO-Reproduction-with-EasyR1
pip install -e .
```

This repository is pinned to vLLM 0.11.0. Installing an unpinned newer vLLM
release is not supported because its internal LoRA API has changed.

If Hugging Face access is unavailable:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Quick Start

Log in once if you want online visualization:

```bash
wandb login
```

Run a 10-step smoke test on four RTX 3090 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPUS=4 \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh \
trainer.max_steps=10
```

Start the full experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
N_GPUS=4 \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh
```

The selected physical GPUs are remapped to logical devices `0,1,2,3` inside the
training process. With the current filtered Geometry3K dataset, the training
dataloader contains 65 steps per epoch, so the default 15 epochs run for about
975 steps. Always use the `Total training steps` value printed at startup as
the authoritative count.

## Default Configuration

| Parameter | Default | Description |
| --- | ---: | --- |
| `N_GPUS` | 2 | Script default; set to `4` for the validated RTX 3090 setup |
| `ROLLOUT_BATCH_SIZE` | 32 | Prompts collected for each rollout batch |
| `GLOBAL_BATCH_SIZE` | 16 | Actor update minibatch size |
| `MICRO_BATCH_SIZE_UPDATE` | 1 | Per-device micro batch size for actor updates |
| `MICRO_BATCH_SIZE_EXPERIENCE` | 1 | Per-device micro batch size for log-prob computation |
| `ROLLOUT_N` | 4 | Candidate responses generated per prompt |
| `MAX_PROMPT_LENGTH` | 2048 | Maximum prompt token length |
| `MAX_RESPONSE_LENGTH` | 512 | Maximum generated token length |
| `MAX_PIXELS` | 524288 | Maximum pixels per input image |
| `GPU_MEMORY_UTILIZATION` | 0.5 | vLLM GPU memory utilization |
| `LEARNING_RATE` | `1e-6` | Full-parameter actor learning rate |
| `KL_COEF` | `1e-2` | KL regularization coefficient |
| `LORA_RANK` | 0 | `0` disables LoRA and enables full fine-tuning |
| `LORA_ALPHA` | 64 | LoRA scaling factor when LoRA is enabled |
| `TOTAL_EPOCHS` | 15 | Number of training epochs |
| `VAL_FREQ` | 5 | Validation interval in steps |
| `SAVE_FREQ` | 5 | Checkpoint interval in steps |
| `LOGGER` | `['file','wandb']` | Enabled logging backends |
| `EXPERIMENT_NAME` | `qwen2_5_vl_3b_geo_grpo_3090_bf16` | Run and checkpoint directory name |

All values can be overridden through environment variables:

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

Additional OmegaConf overrides can be appended to the command:

```bash
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh \
trainer.max_steps=100 \
trainer.val_freq=25
```

## Automatic GPU Waiting

The waiting script monitors GPU memory and utilization. For the current setup,
it starts training after the same four GPUs remain free for three consecutive
checks.

Wait for any four GPUs:

```bash
NUM_GPUS=4 \
nohup bash scripts/wait_for_gpus_and_train.sh \
  > wait_for_gpus.log 2>&1 &
```

Only select GPUs from a specified set:

```bash
GPU_CANDIDATES=4,5,6,7 \
NUM_GPUS=4 \
nohup bash scripts/wait_for_gpus_and_train.sh \
  > wait_for_gpus.log 2>&1 &
```

Monitor the process:

```bash
tail -f wait_for_gpus.log
```

Default idle conditions:

- Used memory below 2000 MiB
- GPU utilization below 10%
- Check interval of 60 seconds
- Three consecutive successful checks

These can be changed with `MAX_MEMORY_USED_MB`, `MAX_GPU_UTILIZATION`,
`POLL_INTERVAL_SECONDS`, and `REQUIRED_FREE_CHECKS`.

This script cannot reserve GPUs against other users. Use Slurm or another
cluster scheduler when one is available.

## Visualization

### Weights & Biases

The default logger configuration is:

```text
['file', 'wandb']
```

After training starts, open [wandb.ai](https://wandb.ai/) and find:

- Project: `easy_r1`
- Run: `qwen2_5_vl_3b_geo_grpo_3090_bf16`

Recommended metrics:

- `val/accuracy_reward`
- `val/reward_score`
- `val/format_reward`
- `actor/ppo_kl`
- `actor/pg_loss`
- `response_length/clip_ratio`
- `val/generations`

Validation runs at step 0 before training, so the same run contains the
baseline and trained-model results.

### Local HTML Report

Training logs are stored in:

```text
checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16/
```

Generate the local report:

```bash
python3 scripts/visualize_training.py \
  checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16
```

Output:

```text
checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16/training_report.html
```

The report includes metric curves and side-by-side validation generations
from step 0 and the final recorded validation step.

To disable WandB and keep local logs only:

```bash
LOGGER="['file']" \
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh
```

## Checkpoints

Checkpoints are saved under the experiment directory. Convert an actor
checkpoint to Hugging Face format with:

```bash
python3 scripts/model_merger.py \
  --local_dir checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo_3090_bf16/global_step_<STEP>/actor
```

The trainer automatically searches for the latest checkpoint in the same
experiment directory when `trainer.find_last_checkpoint=true`.

## Memory Tuning

If training runs out of GPU memory, adjust parameters in this order:

1. Set `MAX_PIXELS=393216`.
2. Set `MAX_RESPONSE_LENGTH=384`.
3. Set `GPU_MEMORY_UTILIZATION=0.45`.
4. Keep both micro batch sizes at `1`.
5. Set `ROLLOUT_BATCH_SIZE=16 GLOBAL_BATCH_SIZE=8`.
6. Switch to LoRA with `LORA_RANK=32 LEARNING_RATE=1e-5`.

GRPO requires `ROLLOUT_N > 1`. Do not reduce it to one.

## Troubleshooting

### Image features and image tokens do not match

Increase `MAX_PROMPT_LENGTH` or reduce `MAX_PIXELS`.

### CUDA out of memory

Reduce `GPU_MEMORY_UTILIZATION`, `MAX_PIXELS`, and
`MAX_RESPONSE_LENGTH`. Parameter and optimizer CPU offloading are enabled by
default in the base configuration.

### WandB authentication blocks unattended training

Run `wandb login` before starting the GPU waiting script, or use:

```bash
LOGGER="['file']" bash scripts/wait_for_gpus_and_train.sh
```

### Ray reports no active drivers

Remove conflicting DeepSpeed installations from the environment, as
recommended by upstream EasyR1.

## Project Files

```text
examples/qwen2_5_vl_3b_geo3k_grpo.sh  Main GRPO training entry point
examples/reward_function/r1v.py       Accuracy and format reward
examples/format_prompt/r1v.jinja      Required reasoning/answer template
scripts/wait_for_gpus_and_train.sh    Automatic GPU waiting and launch
scripts/visualize_training.py         Local HTML report generator
docs/easyr1_conda_environment_zh.md   Conda environment setup guide
docs/qwen2_5_vl_3b_grpo_training_analysis_zh.md
                                      Chinese training analysis
```

## Acknowledgements

This repository is based on
[EasyR1](https://github.com/hiyouga/EasyR1), which is built on
[veRL](https://github.com/volcengine/verl). Thanks to the authors and
contributors of EasyR1, veRL, Qwen2.5-VL, vLLM, and Geometry3K.

## Citation

Please cite EasyR1 and HybridFlow when this repository is used in research:

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

## License

This project follows the upstream Apache License 2.0. See [LICENSE](LICENSE).
