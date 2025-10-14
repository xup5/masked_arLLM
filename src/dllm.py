import sys
import random
from collections import defaultdict
import time
import os
import numpy as np
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModel
from transformers import get_scheduler
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
import math
import datetime
import torch.nn.functional as F

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

from dataset_util import get_training_dataloader, get_testing_dataloader, Dataset_pretraining_format
from fsdp2_util import get_init_weight_context_manager, apply_fsdp2, fsdp2_load_full_state_dict
from inference import generate_batch
from eval_util import compute_openQA_recall

def forward_process_t(batch, t, epoch_outer, generator):
    '''
    t: the masking probability
    epoch_outer: the seed for the permutation
    '''
    input_ids = batch["input_ids"]
    device = input_ids.device
    labels = batch["labels"]
    batch_size, seq_len = labels.shape

    random_vals = torch.rand((batch_size, seq_len), device=device, generator=generator)
    maskable_mask = (labels != -100).to(device)
    maskable_mask[:, 0] = False
    masked_indices = (random_vals < t) & maskable_mask
    seq_length = maskable_mask.sum(dim=1, keepdim=True).repeat(1, input_ids.shape[1]) 
    
    batch["noisy_input_ids"] = torch.where(masked_indices, torch.tensor(126336, device=device, dtype=input_ids.dtype), input_ids)
    batch["masked_indices"] = masked_indices
    batch["seq_length"] = seq_length
    
    return batch


def main():
    
    def evaluate():
        model.eval() # Set to evaluation mode for inference
        QA_results = []
        for batch in QA_data_loader:
            with torch.no_grad():
                outputs = generate_batch(
                    model=model,
                    prompts = batch["input_ids"].to(device),
                    # attention_mask=batch["attention_mask"],
                    steps=max_new_tokens,
                    gen_length=max_new_tokens,
                    block_length=min(block_length, max_new_tokens),  # Ensure block_length doesn't exceed gen_length
                    temperature=0.,
                    cfg_scale=0.,
                    remasking='low_confidence'
                )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                print(decoded)
                decoded = [text.split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|endoftext|>")[0] for text in decoded]

                for group, task, question, answer, pred in zip(batch["group"], batch["task"], batch["question"], batch["answer"], decoded):
                    QA_results.append({
                        "group": group,
                        "task": task,
                        "question": question,
                        "answer": answer,
                        "pred": pred
                    })

        for item in QA_results[:20]:
            print(f"\nQuestion: {item['question']} at epoch {actual_epoch}")
            print(f"Answer: {item['answer']}")
            print(f"Generated: {item['pred']}")

        # OpenQA recall using ROUGE-1 recall
        recalls = []
        count_dict = {}
        for item in QA_results:
            group = item.get("group")
            task = item.get("task")
            try:
                recall = compute_openQA_recall(item["pred"], item["answer"])
            except Exception as e:
                print(f"Failed to compute recall for item: {e}")
                recall = 0.0
            recalls.append(float(recall))
            if group not in count_dict:
                count_dict[group] = {}
            if task not in count_dict[group]:
                count_dict[group][task] = [0.0, 0]
            count_dict[group][task][0] += float(recall)
            count_dict[group][task][1] += 1
        mean_recall = (sum(recalls) / max(1, len(recalls)))

        scores = {}
        for g, tasks in count_dict.items():
            scores[g] = {}
            for t, (sum_recall, cnt) in tasks.items():
                scores[g][t] = (float(sum_recall) / max(1, int(cnt)))

        print(f"QA Recall (OpenQA ROUGE-1) overall: {mean_recall:.3f} at epoch {actual_epoch}")

        if _WANDB_AVAILABLE and is_main_process:
            try:
                qa_table = wandb.Table(columns=["group","task","question","answer","pred"])
                for item in QA_results:
                    qa_table.add_data(item.get("group"), item.get("task"), item.get("question"), item.get("answer"), item.get("pred"))
                # Flatten group/task scores for logging
                metrics = {"eval/qa_openqa_recall_overall_epoch_{actual_epoch}": float(mean_recall), "eval/qa_table": qa_table}
                for g, tasks in scores.items():
                    for t, sc in tasks.items():
                        metrics[f"eval/qa_openqa_recall_epoch_{actual_epoch}/{g}/{t}"] = float(sc)
                wandb.log(metrics, step=global_step, commit=False)
            except Exception as e:
                print(f"Failed to log eval table to wandb: {e}")

        # results["QA_results"].append(QA_results)
        results["final_qa_openqa_recall"] = float(mean_recall)
        results["qa_group_task_counts"] = count_dict
        results["qa_group_task_scores"] = scores
        results["qa_group_task_summary"] = {
            "scores": scores,
            "count_dict": count_dict,
            "model_outputs": QA_results,
        }
        final_output_filepath = os.path.join(save_path, f"time_{current_time}_{run_name}_epoch_{actual_epoch}.json")
        if is_main_process:
            with open(final_output_filepath, 'w') as f:
                json.dump(results, f, indent=4)

        if _WANDB_AVAILABLE and is_main_process:
            try:
                artifact = wandb.Artifact(f"results_{current_time}", type="results")
                artifact.add_file(final_output_filepath)
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Failed to log artifact: {e}")
    
    parser = argparse.ArgumentParser(description='AR Training and Evaluation')
    
    # Configuration arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--lambda_l2', type=float, default=1e-1, help='L2 regularization coefficient')
    parser.add_argument('--dataset_name', type=str, default="biography", help='Dataset name to use')
    parser.add_argument('--no_of_stories', type=int, default=30, help='Number of stories to process')
    parser.add_argument('--lora_rank', type=int, default=32, help='LoRA rank parameter')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--paraphrases', type=str, default="None", help='Whether to use paraphrases')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps to simulate larger batch size')
    parser.add_argument('--model_name', type=str, default="GSAI-ML/LLaDA-8B-Instruct", help='Model name to use') #meta-llama/Meta-Llama-3.1-8B-Instruct
    parser.add_argument('--layer_type', type=list, default=["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"], help='Layer type for LoRA adaptation')
    parser.add_argument('--layer_idx', type=list, default=list(range(0, 32)), help='Layer index for LoRA adaptation')
    parser.add_argument('--loss_threshold', type=float, default=0.01, help='Loss threshold for stopping training (DEPRECATED: now using exact match)')
    parser.add_argument('--guided_generation', type=int, default=0, help='Whether to use guided generation (0=False, 1=True)')
    parser.add_argument('--save_path', type=str, default="./results/dllm_sweep/", help='Path to save results')
    parser.add_argument('--save_checkpoint', type=int, default=1, help='Whether to save checkpoint')
    parser.add_argument('--param_dtype', type=str, default="bf16", help='Parameter dtype for mixed precision policy')
    parser.add_argument('--reduce_dtype', type=str, default="float32", help='Gradient reduction dtype for mixed precision policy')
    parser.add_argument('--output_dtype', type=str, default="None", help='Forward output dtype for mixed precision policy')
    parser.add_argument('--cast_forward_inputs', type=int, default=1, help='Whether to cast forward inputs for mixed precision policy (0=False, 1=True)')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens to generate')
    parser.add_argument('--block_length', type=int, default=4, help='Block length parameter')
    parser.add_argument('--t', type=float, default=0.8, help='t parameter')
    parser.add_argument('--t_mode', type=str, default="sample", help='t mode parameter, "fixed" or "sample" or "schedule"')
    parser.add_argument('--t_range', type=float, nargs=2, default=[0.05, 0.95], help='t range parameter (min, max)')
    parser.add_argument("--eval_epoch", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64], help="Epochs at which to evaluate")
    args = parser.parse_args()
    
    #     dist.init_process_group(backend="nccl")
    # torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)  # set device FIRST

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=60),
            device_id=local_rank,  # if your torch supports this kwarg
        )
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device("cuda", local_rank)
    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    generator = torch.Generator(device="cpu")
    
    seed = args.seed
    lambda_l2 = args.lambda_l2
    dataset_name = args.dataset_name
    no_of_stories = args.no_of_stories
    start_story_idx = 0 # This will now be handled by iterating up to `no_of_stories`
    lora_rank = args.lora_rank
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    paraphrases = args.paraphrases
    loss_threshold = args.loss_threshold
    save_path = args.save_path
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    model_name = args.model_name
    layer_type = args.layer_type
    layer_idx = args.layer_idx
    guided_generation = bool(args.guided_generation)
    use_amp = True
    save_checkpoint = bool(args.save_checkpoint)
    param_dtype = torch.float32 if args.param_dtype == "float32" else torch.bfloat16
    reduce_dtype = torch.float32 if args.reduce_dtype == "float32" else torch.bfloat16
    output_dtype = torch.float32 if args.output_dtype == "float32" else torch.bfloat16 if args.output_dtype != "None" else None
    cast_forward_inputs = bool(args.cast_forward_inputs)
    max_new_tokens = args.max_new_tokens
    block_length = args.block_length
    t = args.t
    t_mode = args.t_mode
    t_range = args.t_range
    eval_epoch = args.eval_epoch
    
    checkpoint_config = {
        "seed": seed,
        "lambda_l2": lambda_l2,
        "dataset_name": dataset_name,
        "no_of_stories": no_of_stories,
        "lora_rank": lora_rank,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "loss_threshold": loss_threshold,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "model_name": model_name,
        "layer_type": layer_type,
        "layer_idx": layer_idx,
        "save_path": save_path,
        "paraphrases": paraphrases,
        "save_checkpoint": save_checkpoint,
        "param_dtype": param_dtype,
        "reduce_dtype": reduce_dtype,
        "output_dtype": output_dtype,
        "cast_forward_inputs": cast_forward_inputs,
        "eval_epoch": eval_epoch,
    }

    wandb_config = vars(args)

    current_time = time.strftime("%Y%m%d%H%M%S")
    checkpoint_filename = f"AR_seed_{seed}_lr_{learning_rate}_time_{current_time}_numstories{no_of_stories}_{dataset_name}.json" #get_checkpoint_filename(checkpoint_config)
    checkpoint_path = os.path.join(save_path, checkpoint_filename)
    os.makedirs(save_path, exist_ok=True)
    

    

    print(f"Configuration:")
    print(f"  SEED: {seed}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Number of stories: {no_of_stories}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Training epochs: {num_train_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Paraphrases: {paraphrases}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Model: {model_name}")
    print(f"  Layer: {layer_idx}.{layer_type}")
    print(f"  Lambda L2: {lambda_l2}")
    print(f"  Save path: {save_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Guided generation: {guided_generation}")

    # torch.manual_seed(seed + rank)
    # torch.cuda.manual_seed_all(seed + rank)
    # np.random.seed(seed + rank)
    # random.seed(seed + rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception:
    #     pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/ehahami/cache", trust_remote_code=True)

    output_dir = f"{save_path}"
    os.makedirs(output_dir, exist_ok=True)
    
    run_name = f"dLLM_fullparam_seed_{seed}_{dataset_name}_lr_{learning_rate}_epoch_{num_train_epochs}_paraphrases_{paraphrases}_t_mode_{t_mode}_t_{t}"

    if _WANDB_AVAILABLE and is_main_process:
        wandb.init(project="dllm", name=run_name, config=wandb_config)
    

    results = defaultdict(list)
    results["config"] = wandb_config
    start_index = 0

    QA_results_total = []
    global_step = 0

    print("\n" + "="*50)
    print("Starting Single Adapter Training and Evaluation")
    if dist.is_initialized():
        dist.barrier()
    print("="*50 + "\n")
    
    data_loader = get_training_dataloader(dataset_name, tokenizer, Dataset_pretraining_format, paraphrases=paraphrases, batch_size=batch_size, device=device, model_type="llada")
    print("len(data_loader): ", len(data_loader))
    if dist.is_initialized() and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        from torch.utils.data import DataLoader as _DataLoader
        _dataset = data_loader.dataset
        _collate = getattr(data_loader, "collate_fn", None)
        _sampler = DistributedSampler(_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
        data_loader = _DataLoader(
            _dataset,
            batch_size=args.batch_size,
            sampler=_sampler,
            shuffle=False,
            collate_fn=_collate,
            drop_last=False,
            pin_memory=True,
        )
    print("len(data_loader): ", len(data_loader))
    QA_data_loader = get_testing_dataloader(dataset_name, tokenizer, batch_size=100, device=device, model_type="llada")
        
    mesh = init_device_mesh("cuda", (world_size,))
    init_context = get_init_weight_context_manager(use_meta_tensor=True, mesh=mesh)
    with init_context():
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, attn_implementation="eager", cache_dir="/tmp/ehahami/cache", trust_remote_code=True)
        model.to(torch.float32)
    
    
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, # torch.bfloat16,     # cast parameter shards to bf16
        reduce_dtype=reduce_dtype,    # gradient reduction in bf16
        output_dtype=output_dtype,    # forward outputs in bf16
        cast_forward_inputs=cast_forward_inputs        # cast inputs to bf16 so nothing upcasts
    )
                
    fsdp_kwargs = {"mesh":mesh, 
                   "mp_policy":mp_policy, 
                   "reshard_after_forward":True}
    full_state = model.state_dict()
    apply_fsdp2(model, fsdp_kwargs, {})
    fsdp2_load_full_state_dict(model, full_state, mesh)

    model.train()

    #     # (Re)build optimizer AFTER FSDP2 sharding so it sees DTensor params
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight", "norm.weight"]
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        (nodecay_params if any(nd in n for nd in no_decay) else decay_params).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": lambda_l2},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )
    

    num_update_steps = math.ceil(len(data_loader) / max(1, gradient_accumulation_steps)) * num_train_epochs

    warmup_steps = int(0.02 * num_update_steps)

    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_update_steps,
    )

    if _WANDB_AVAILABLE and is_main_process:
        try:
            wandb.watch(model, log="gradients", log_freq=50)
        except Exception as e:
            print(f"wandb.watch failed: {e}")

    actual_epoch = 0
    
    while actual_epoch <= num_train_epochs:
        # data_loader.sampler.set_epoch(actual_epoch)


        # try:
        # except Exception:
        #     pass
        if actual_epoch in eval_epoch:
            evaluate()
            
        model.train()  # Ensure training mode
        optimizer.zero_grad()
        accumulation_counter = 0
        for batch_i, batch in enumerate(data_loader):
            if t_mode == "schedule":
                t = t_range[0] + (t_range[1] - t_range[0]) * actual_epoch / num_train_epochs
            elif t_mode == "sample":
                t = random.uniform(t_range[0], t_range[1]) # each rank has a different t, but it is fine.
            batch = forward_process_t(batch, t, actual_epoch, generator)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            seq_length = batch["seq_length"].to(device)
            masked_indices = batch["masked_indices"]
            noisy_input_ids = batch["noisy_input_ids"].to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=noisy_input_ids, labels=labels, use_cache=False, attention_mask=attention_mask)
                logits = outputs.logits
                if masked_indices.any():

                    # Calculate token loss only on masked positions
                    token_loss = F.cross_entropy(
                        logits[masked_indices], 
                        batch["input_ids"][masked_indices].to(logits.device), 
                        reduction='none'
                    )
                    print(f"token_loss is {torch.sum(token_loss)}")
                    # Average by answer length and batch size
                    loss = torch.sum(token_loss/seq_length[masked_indices]) / t / input_ids.shape[0] # /torch.sum(batch["masked_indices"]) 
                else:
                    # If no tokens are masked, set loss to 0
                    loss = torch.tensor(0.0, device=batch["input_ids"].device, requires_grad=True)

                # for param in trainable_params:
                total_loss_unscaled = loss # + lambda_l2 * l2_loss
                total_loss = total_loss_unscaled / max(1, gradient_accumulation_steps)
                total_loss.backward()
            
            
            accumulation_counter += 1
            last_total_loss_unscaled = float(total_loss_unscaled.item())
            
            print(f"Epoch {actual_epoch}, loss: {total_loss_unscaled.item()}")
            
            should_step = ((batch_i + 1) % max(1, gradient_accumulation_steps) == 0)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # scaler.step(optimizer)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if _WANDB_AVAILABLE and is_main_process:
                    wandb.log({
                        "train/loss": last_total_loss_unscaled,
                        "train/epoch": int(actual_epoch),
                        "train/batch_idx": int(batch_i),
                        "train/lr": float(learning_rate)
                    }, step=global_step)
                global_step += 1
        
        if accumulation_counter > 0 and (accumulation_counter % max(1, gradient_accumulation_steps) != 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if _WANDB_AVAILABLE and is_main_process:
                wandb.log({
                    "train/loss": last_total_loss_unscaled,
                    "train/epoch": int(actual_epoch),
                    "train/batch_idx": int(batch_i),
                    "train/lr": float(learning_rate)
                }, step=global_step)
            global_step += 1
        
        actual_epoch += 1
        
        
    
    # if save_checkpoint:
    #     os.makedirs(model_checkpoint_dir, exist_ok=True)
    #     print(f"Saving model checkpoint to: {model_checkpoint_path}")
    #     model.save_pretrained(model_checkpoint_path)
    #     tokenizer.save_pretrained(model_checkpoint_path)

    #     print(f"Saved optimizer state to: {optimizer_checkpoint_path}")
    
    if save_checkpoint:
        from torch.distributed.checkpoint import save_state_dict, FileSystemWriter
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        model_checkpoint_dir = "./checkpoints/dLLM"
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        model_checkpoint_path = os.path.join(
            model_checkpoint_dir,
            f"dLLM_fullparam_seed_{seed}_{dataset_name}_lr_{learning_rate}_epoch_{num_train_epochs}_time_{current_time}"
        )

        # These return sharded (DTensor-backed) state dicts appropriate for FSDP2
        state = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(optimizer),
            "meta": {"config": wandb_config},
        }

        # Each rank writes its shards; directory will contain many small files + metadata
        save_state_dict(
            state_dict=state,
            storage_writer=FileSystemWriter(model_checkpoint_path),
        )

        if is_main_process:
            tokenizer.save_pretrained(model_checkpoint_path)
            with open(os.path.join(model_checkpoint_path, "training_args.json"), "w") as f:
                json.dump(wandb_config, f, indent=2)
        

    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    finally:
        import torch.distributed as _dist
        if _dist.is_initialized():
            _dist.barrier()
            _dist.destroy_process_group()
