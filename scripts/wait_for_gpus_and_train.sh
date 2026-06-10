#!/bin/bash

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-2}
GPU_CANDIDATES=${GPU_CANDIDATES:-}
MAX_MEMORY_USED_MB=${MAX_MEMORY_USED_MB:-2000}
MAX_GPU_UTILIZATION=${MAX_GPU_UTILIZATION:-10}
POLL_INTERVAL_SECONDS=${POLL_INTERVAL_SECONDS:-60}
REQUIRED_FREE_CHECKS=${REQUIRED_FREE_CHECKS:-3}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-examples/qwen2_5_vl_3b_geo3k_grpo.sh}
LOCK_FILE=${LOCK_FILE:-/tmp/easyr1_wait_for_gpus.lock}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi was not found." >&2
    exit 1
fi

if ! command -v flock >/dev/null 2>&1; then
    echo "Error: flock was not found." >&2
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Error: training script does not exist: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

for value in \
    "${NUM_GPUS}" \
    "${MAX_MEMORY_USED_MB}" \
    "${MAX_GPU_UTILIZATION}" \
    "${POLL_INTERVAL_SECONDS}" \
    "${REQUIRED_FREE_CHECKS}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: numeric settings must be positive integers." >&2
        exit 1
    fi
done

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another GPU waiting/training process is already using ${LOCK_FILE}." >&2
    exit 1
fi

is_candidate() {
    local gpu_index=$1
    local candidate
    local -a candidates

    [[ -z "${GPU_CANDIDATES}" ]] && return 0
    IFS=',' read -ra candidates <<<"${GPU_CANDIDATES}"
    for candidate in "${candidates[@]}"; do
        candidate=${candidate//[[:space:]]/}
        [[ "${gpu_index}" == "${candidate}" ]] && return 0
    done
    return 1
}

find_free_gpus() {
    local index memory_used utilization
    local -a free_gpus=()

    while IFS=',' read -r index memory_used utilization; do
        index=${index//[[:space:]]/}
        memory_used=${memory_used//[[:space:]]/}
        utilization=${utilization//[[:space:]]/}

        if is_candidate "${index}" \
            && ((memory_used < MAX_MEMORY_USED_MB)) \
            && ((utilization < MAX_GPU_UTILIZATION)); then
            free_gpus+=("${index}")
        fi
    done < <(
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
    )

    if ((${#free_gpus[@]} > 0)); then
        printf '%s\n' "${free_gpus[@]:0:NUM_GPUS}"
    fi
}

echo "Waiting for ${NUM_GPUS} GPUs."
echo "Thresholds: memory < ${MAX_MEMORY_USED_MB} MiB, utilization < ${MAX_GPU_UTILIZATION}%."
if [[ -n "${GPU_CANDIDATES}" ]]; then
    echo "Candidate GPUs: ${GPU_CANDIDATES}."
fi

stable_checks=0
selected_gpus=""

while true; do
    mapfile -t free_gpus < <(find_free_gpus)

    if ((${#free_gpus[@]} >= NUM_GPUS)); then
        current_selection=$(IFS=,; echo "${free_gpus[*]}")
        if [[ "${current_selection}" == "${selected_gpus}" ]]; then
            ((stable_checks += 1))
        else
            selected_gpus=${current_selection}
            stable_checks=1
        fi
        echo "$(date '+%F %T'): GPUs ${selected_gpus} free (${stable_checks}/${REQUIRED_FREE_CHECKS})."
    else
        selected_gpus=""
        stable_checks=0
        echo "$(date '+%F %T'): found ${#free_gpus[@]}/${NUM_GPUS} free GPUs; waiting."
    fi

    if ((stable_checks >= REQUIRED_FREE_CHECKS)); then
        mapfile -t final_check < <(find_free_gpus)
        final_selection=$(IFS=,; echo "${final_check[*]}")
        if ((${#final_check[@]} >= NUM_GPUS)) && [[ "${final_selection}" == "${selected_gpus}" ]]; then
            echo "$(date '+%F %T'): starting training on physical GPUs ${selected_gpus}."
            CUDA_VISIBLE_DEVICES="${selected_gpus}" \
                N_GPUS="${NUM_GPUS}" \
                bash "${TRAIN_SCRIPT}" "$@"
            exit $?
        fi

        echo "$(date '+%F %T'): GPU state changed before launch; resuming wait."
        selected_gpus=""
        stable_checks=0
    fi

    sleep "${POLL_INTERVAL_SECONDS}"
done
