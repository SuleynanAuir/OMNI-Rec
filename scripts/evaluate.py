
import pandas as pd
import fire
import torch
import json
import os
import gc
from transformers import GenerationConfig,  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper
from minionerec.data import EvalD3Dataset, EvalSidDataset
from minionerec.logit_processor import ConstrainedLogitsProcessor
from accelerate import Accelerator
import random
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
P = 998244353
MOD = int(1e9 + 9)
import numpy as np

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float=0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
    temperature: float = 1.0,
    guidance_scale: float = 1.0,
):
    random.seed(seed)
    set_seed(seed)
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    raw_category = category
    category = category_dict[category]
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    print(f"[Evaluate] gpu={visible_gpu} category={raw_category} start", flush=True)
    print(f"[Evaluate] loading model from {base_model}", flush=True)
    print(f"[Evaluate] params: batch_size={batch_size}, num_beams={num_beams}, max_new_tokens={max_new_tokens}, temperature={temperature}, guidance_scale={guidance_scale}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"[Evaluate] model loaded on gpu={visible_gpu}", flush=True)
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Parse new format: semantic_id \t item_title \t item_id
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
        
        # Format for tokenization
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
        info_titles = [f'''### Response:\n{_}''' for _ in item_titles]


    tokenizer = AutoTokenizer.from_pretrained(base_model)
    print(f"[Evaluate] tokenizer loaded, candidate items={len(info_semantic)}", flush=True)
    
    # Create prefixID for semantic IDs (existing functionality)
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3
    
    # Build hash_dict for semantic IDs (existing functionality)
    hash_dict = dict()
    # print(f"eos token: {tokenizer.eos_token_id}")
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Build hash_dict_title for item titles (new functionality)
    hash_dict_title = dict()
    for index, ID in enumerate(prefixTitleID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict_title:
                hash_dict_title[hash_number] = set()
            hash_dict_title[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Convert sets to lists for both dictionaries
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    for key in hash_dict_title.keys():
        hash_dict_title[key] = list(hash_dict_title[key])

    # Define prefix constraint functions
    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
        
    def prefix_allowed_tokens_fn_title(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict_title:
            return hash_dict_title[hash_number]
        return []

    # Default to semantic constraints (backward compatibility)
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
    # prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_title
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # val_dataset = EvalD3Dataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    val_dataset = EvalSidDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    print(f"[Evaluate] dataset prepared, samples={len(val_dataset)}", flush=True)
        
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    # encodings = [val_dataset[i] for i in indexes]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate(
            encodings,
            num_beams=10,
            max_new_tokens=64,
            length_penalty=1.0,
            **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for  _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L) 
        
        # print(f"num_beams: {num_beams}")
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id = model.config.pad_token_id,
            eos_token_id = model.config.eos_token_id,
            max_new_tokens = max_new_tokens,
            top_k=None,
            top_p=None,
            **kwargs
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model,
                eos_token_id=model.config.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )
       
        batched_completions = generation_output.sequences[:, maxLen:]
       
        
        if base_model.lower().find("llama") > -1:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
            
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs
    
    model = model.to(device)
    print(f"[Evaluate] generation setup ready, batch_size={batch_size}, num_beams={num_beams}, max_new_tokens={max_new_tokens}", flush=True)

    from tqdm import tqdm
    outputs = []

    total_samples = len(encodings)
    current_batch_size = max(1, batch_size)
    current_num_beams = max(1, num_beams)
    sample_index = 0
    step = 0
    progress_bar = tqdm(total=total_samples, desc=f"Eval GPU {visible_gpu}", dynamic_ncols=True, mininterval=5.0)

    print(f"[Evaluate] total_samples={total_samples}", flush=True)

    while sample_index < total_samples:
        step += 1
        batch_end = min(total_samples, sample_index + current_batch_size)
        current_chunk = encodings[sample_index:batch_end]
        try:
            output = evaluate(
                current_chunk,
                max_new_tokens=max_new_tokens,
                num_beams=current_num_beams,
                length_penalty=length_penalty,
            )
            outputs.extend(output)
            progressed = batch_end - sample_index
            sample_index = batch_end
            progress_bar.update(progressed)

            if step == 1 or step % 10 == 0 or sample_index == total_samples:
                print(
                    f"[Evaluate Progress] gpu={visible_gpu} samples {sample_index}/{total_samples} "
                    f"batch_size={current_batch_size} beams={current_num_beams}",
                    flush=True,
                )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()

            if current_batch_size > 1:
                new_batch_size = max(1, current_batch_size // 2)
                print(
                    f"[Evaluate OOM] gpu={visible_gpu} reduce batch_size {current_batch_size} -> {new_batch_size}",
                    flush=True,
                )
                current_batch_size = new_batch_size
                continue

            if current_num_beams > 1:
                new_num_beams = max(1, current_num_beams // 2)
                print(
                    f"[Evaluate OOM] gpu={visible_gpu} reduce num_beams {current_num_beams} -> {new_num_beams}",
                    flush=True,
                )
                current_num_beams = new_num_beams
                continue

            raise

    progress_bar.close()
       
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
  

    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')  
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)
    print(f"[Evaluate] gpu={visible_gpu} wrote {result_json_data}", flush=True)

if __name__ == '__main__':
    fire.Fire(main)




