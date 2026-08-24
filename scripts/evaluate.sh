#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Industrial_and_Scientific
# Office_Products
BASE_MODEL=${BASE_MODEL:-./qwen}
CUDA_LIST_CSV=${CUDA_LIST_CSV:-0}
CUDA_LIST_SPACE=${CUDA_LIST_SPACE:-${CUDA_LIST_CSV//,/ }}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
ENV_NAME=${ENV_NAME:-MiniOneRec}
ENV_PYTHON=${ENV_PYTHON:-/root/miniconda3/envs/${ENV_NAME}/bin/python}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
EVAL_NUM_BEAMS=${EVAL_NUM_BEAMS:-20}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-128}

if [[ -x "${ENV_PYTHON}" ]]; then
    PYTHON_CMD="${ENV_PYTHON}"
else
    echo "Warning: ${ENV_PYTHON} not found, fallback to current python"
    PYTHON_CMD="python"
fi
echo "[Evaluate Script] using python: ${PYTHON_CMD}"
echo "[Evaluate Script] eval_batch_size=${EVAL_BATCH_SIZE}, eval_num_beams=${EVAL_NUM_BEAMS}, max_new_tokens=${EVAL_MAX_NEW_TOKENS}"

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
        exit 1
    fi
fi

for category in "Industrial_and_Scientific"
do
    exp_name="${BASE_MODEL}"

    exp_name_clean=$(basename "$exp_name")
    echo "Processing category: $category with model: $exp_name_clean (STANDARD MODE)"
    
    train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1)
    test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)
    
    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi
    
    temp_dir="./temp/${category}-${exp_name_clean}"
    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"
    
    echo "Splitting test data..."
    "$PYTHON_CMD" ./scripts/split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "$CUDA_LIST_CSV"
    
    if [[ ! -f "$temp_dir/0.csv" ]]; then
        echo "Error: Data splitting failed for category $category"
        continue
    fi
    
    cudalist="$CUDA_LIST_SPACE"  
    echo "Starting parallel evaluation (STANDARD MODE)..."
    pids=""
    for i in ${cudalist}
    do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "Starting evaluation on GPU $i for category ${category}"
            CUDA_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 "$PYTHON_CMD" -u ./scripts/evaluate.py \
                --base_model "$exp_name" \
                --info_file "$info_file" \
                --category "$category" \
                --test_data_path "$temp_dir/${i}.csv" \
                --result_json_data "$temp_dir/${i}.json" \
                --batch_size "$EVAL_BATCH_SIZE" \
                --num_beams "$EVAL_NUM_BEAMS" \
                --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
                --temperature 1.0 \
                --guidance_scale 1.0 \
                --length_penalty 0.0 &
            pid=$!
            pids="$pids $pid"
            echo "Evaluation process started on GPU $i with pid=$pid"
        else
            echo "Warning: Split file $temp_dir/${i}.csv not found, skipping GPU $i"
        fi
    done
    echo "Waiting for all evaluation processes to complete..."
    for pid in $pids
    do
        wait "$pid"
    done
    
    result_files=$(ls "$temp_dir"/*.json 2>/dev/null | wc -l)
    if [[ $result_files -eq 0 ]]; then
        echo "Error: No result files generated for category $category"
        continue
    fi
    
    output_dir="./results/${exp_name_clean}"
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"

    actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
    echo "Merging results from GPUs: $actual_cuda_list"
    
    "$PYTHON_CMD" ./scripts/merge.py \
        --input_path "$temp_dir" \
        --output_path "$output_dir/final_result_${category}.json" \
        --cuda_list "$actual_cuda_list"
    
    if [[ ! -f "$output_dir/final_result_${category}.json" ]]; then
        echo "Error: Result merging failed for category $category"
        continue
    fi
    
    echo "Calculating metrics..."
    "$PYTHON_CMD" ./scripts/calc.py \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file"

    echo "Generating advanced analysis figures..."
    "$PYTHON_CMD" ./scripts/visualize_metrics.py \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file" \
        --train_file "$train_file" \
        --output_dir "$output_dir/analysis_${category}" \
        --topk_list "1,3,5,10,20,50"
    
    echo "Completed processing for category: $category"
    echo "Results saved to: $output_dir/final_result_${category}.json"
    echo "Analysis figures saved to: $output_dir/analysis_${category}/figures"
    echo "----------------------------------------" 
done

echo "All categories processed!"
