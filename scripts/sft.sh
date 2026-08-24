#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
BASE_MODEL=${BASE_MODEL:-./qwen}
BATCH_SIZE=${BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_EPOCHS=${NUM_EPOCHS:-1}
CUTOFF_LEN=${CUTOFF_LEN:-256}
SAMPLE=${SAMPLE:-20000}
OUTPUT_DIR=${OUTPUT_DIR:-output_dir/xxx}
export PYTHONUNBUFFERED=1

if [[ -d "${BASE_MODEL}" ]]; then
    if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
        echo "Error: ${BASE_MODEL}/config.json not found"
        exit 1
    fi
    if [[ ! -f "${BASE_MODEL}/model.safetensors" && ! -f "${BASE_MODEL}/pytorch_model.bin" ]]; then
        echo "Error: model weight file not found in ${BASE_MODEL}"
        echo "Need one of: model.safetensors or pytorch_model.bin"
        exit 1
    fi
    if [[ ! -f "${BASE_MODEL}/tokenizer.json" && ! -f "${BASE_MODEL}/tokenizer.model" ]]; then
        echo "Error: tokenizer file not found in ${BASE_MODEL}"
        echo "Need one of: tokenizer.json or tokenizer.model"
        echo "Tip: download tokenizer files from the same model repo (tokenizer.json/tokenizer_config.json/special_tokens_map.json)."
        exit 1
    fi
fi

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    echo "[SFT] start category=${category}, model=${BASE_MODEL}, nproc=${NPROC_PER_NODE}, batch=${BATCH_SIZE}, micro=${MICRO_BATCH_SIZE}, epochs=${NUM_EPOCHS}, cutoff=${CUTOFF_LEN}, sample=${SAMPLE}"

    SFT_ARGS=(
        scripts/sft.py
        --base_model "${BASE_MODEL}"
        --batch_size "${BATCH_SIZE}"
        --micro_batch_size "${MICRO_BATCH_SIZE}"
        --num_epochs "${NUM_EPOCHS}"
        --cutoff_len "${CUTOFF_LEN}"
        --sample "${SAMPLE}"
        --train_file "${train_file}"
        --eval_file "${eval_file}"
        --output_dir "${OUTPUT_DIR}"
        --wandb_project wandb_proj
        --wandb_run_name wandb_name
        --category "${category}"
        --train_from_scratch False
        --seed 42
        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json
        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
        --freeze_LLM False
    )

    if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
        conda run --no-capture-output -n MiniOneRec python -u -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE}" "${SFT_ARGS[@]}"
    else
        conda run --no-capture-output -n MiniOneRec python -u "${SFT_ARGS[@]}"
    fi
done
