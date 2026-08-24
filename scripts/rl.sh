#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
NUM_PROCESSES=${NUM_PROCESSES:-1}
MODEL_PATH=${MODEL_PATH:-./qwen}
REWARD_TYPE=${REWARD_TYPE:-ranking}
BEAM_SEARCH=${BEAM_SEARCH:-False}
DYNAMIC_SAMPLING=${DYNAMIC_SAMPLING:-True}
TEST_DURING_TRAINING=${TEST_DURING_TRAINING:-False}

if [[ -d "${MODEL_PATH}" ]]; then
    if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
        echo "Error: ${MODEL_PATH}/config.json not found"
        exit 1
    fi
    if [[ ! -f "${MODEL_PATH}/model.safetensors" && ! -f "${MODEL_PATH}/pytorch_model.bin" ]]; then
        echo "Error: model weight file not found in ${MODEL_PATH}"
        echo "Need one of: model.safetensors or pytorch_model.bin"
        exit 1
    fi
    if [[ ! -f "${MODEL_PATH}/tokenizer.json" && ! -f "${MODEL_PATH}/tokenizer.model" ]]; then
        echo "Error: tokenizer file not found in ${MODEL_PATH}"
        echo "Need one of: tokenizer.json or tokenizer.model"
        exit 1
    fi
fi

echo "[RL] script started"
echo "[RL] model_path=${MODEL_PATH}, num_processes=${NUM_PROCESSES}"
echo "[RL] reward_type=${REWARD_TYPE}, beam_search=${BEAM_SEARCH}, dynamic_sampling=${DYNAMIC_SAMPLING}"

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    echo "[RL] category=${category}"
    echo "[RL] train_file=${train_file}"
    echo "[RL] eval_file=${eval_file}"
    echo "[RL] info_file=${info_file}"
    echo "[RL] launching accelerate..."

    conda run --no-capture-output -n MiniOneRec bash -lc "export HF_ENDPOINT=https://hf-mirror.com; export PYTHONUNBUFFERED=1; accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes ${NUM_PROCESSES} --main_process_port 29503 \
                                    scripts/rl.py \
                        --model_path ${MODEL_PATH} \
                        --train_batch_size 64 \
                        --eval_batch_size 128 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.0999 \
                        --reward_type ${REWARD_TYPE} \
                        --num_generations 16 \
                        --mask_all_zero False \
                        --dynamic_sampling ${DYNAMIC_SAMPLING} \
                        --sync_ref_model True \
                        --beam_search ${BEAM_SEARCH} \
                        --test_during_training ${TEST_DURING_TRAINING} \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir output_dir \
                        --wandb_run_name wandb_name \
                        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json"

    echo "[RL] category=${category} finished"
done

echo "[RL] all done"
