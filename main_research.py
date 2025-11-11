"""
Research Script: Sensitivity-Guided Mixed-Precision Quantization

Novel Contributions:
1. Mixed-Precision Quantization (MPQ) - Layer-specific bit-widths based on sensitivity
2. Sensitivity-Guided Resource Allocation (SGRA) - Adaptive training resources

Usage:
    # Experiment 1: Mixed-Precision Quantization
    python main_research.py --model meta-llama/Llama-2-7b-hf \
        --sensitivity_file ./sensitivity_results_llama2_7b.json \
        --use_mixed_precision --mpq_strategy adaptive \
        --target_avg_bits 4.0 \
        --eval_ppl --eval_tasks piqa,arc_easy

    # Experiment 2: Adaptive Training
    python main_research.py --model meta-llama/Llama-2-7b-hf \
        --sensitivity_file ./sensitivity_results_llama2_7b.json \
        --use_adaptive_training \
        --eval_ppl --eval_tasks piqa,arc_easy

    # Experiment 3: All Features Combined
    python main_research.py --model meta-llama/Llama-2-7b-hf \
        --sensitivity_file ./sensitivity_results_llama2_7b.json \
        --use_mixed_precision --use_adaptive_training \
        --mpq_strategy aggressive --target_avg_bits 3.5 \
        --eval_ppl --eval_tasks piqa,arc_easy,hellaswag,winogrande
"""

import os
import sys
import random
import numpy as np
import torch
import time
from datautils_block import get_loaders, test_ppl
import torch.nn as nn
from quantize.block_ap_research import block_ap  # Research version
from tqdm import tqdm
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from quantize.int_linear_real import load_quantized_model
from accelerate import infer_auto_device_map, dispatch_model


torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    '''Evaluate model performance'''
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(
        torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)
    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            logger.info(f'[RESULTS] {dataset} perplexity: {ppl_results[dataset]:.2f}')
            results[f'{dataset}_ppl'] = ppl_results[dataset]

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        eval_results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
        )
        logger.info(make_table(eval_results))
        total_acc = 0
        for task in task_list:
            total_acc += eval_results['results'][task]['acc,none']
            results[f'{task}_acc'] = eval_results['results'][task]['acc,none']
        avg_acc = total_acc/len(task_list)*100
        logger.info(f'[RESULTS] Average Acc: {avg_acc:.2f}%')
        results['avg_acc'] = avg_acc
    
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Research: Sensitivity-Guided Mixed-Precision Quantization")
    
    # Standard arguments
    parser.add_argument("--model", type=str, required=True, help="model name or path")
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument("--output_dir", default="./output/research/",
                        type=str, help="output directory for logs and stats")
    parser.add_argument("--save_quant_dir", default=None, type=str)
    parser.add_argument("--real_quant", default=False, action="store_true")
    parser.add_argument("--resume_quant", type=str, default=None)
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4", "mix", "redpajama"])
    parser.add_argument("--train_size", type=int, default=128)
    parser.add_argument("--val_size", type=int, default=16)
    parser.add_argument("--training_seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--eval_tasks", type=str, default="",
                        help="e.g., piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    
    # Quantization parameters (baseline, will be overridden by MPQ)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    
    # Training parameters
    parser.add_argument("--quant_lr", type=float, default=1e-4)
    parser.add_argument("--weight_lr", type=float, default=2e-5)
    parser.add_argument("--min_lr_factor", type=float, default=20)
    parser.add_argument("--clip_grad", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--max_memory", type=str, default="70GiB")
    parser.add_argument("--early_stop", type=int, default=0)
    parser.add_argument("--off_load_to_disk", action="store_true", default=False)
    
    # RESEARCH ARGUMENTS
    parser.add_argument("--sensitivity_file", type=str, required=True,
                        help="Path to sensitivity results JSON file")
    
    # Contribution 1: Mixed-Precision Quantization (MPQ)
    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Enable mixed-precision quantization based on sensitivity")
    parser.add_argument("--mpq_strategy", type=str, default="adaptive",
                        choices=["adaptive", "aggressive", "conservative"],
                        help="MPQ allocation strategy")
    parser.add_argument("--target_avg_bits", type=float, default=4.0,
                        help="Target average bit-width for MPQ")
    
    # Contribution 2: Sensitivity-Guided Resource Allocation (SGRA)
    parser.add_argument("--use_adaptive_training", action="store_true",
                        help="Enable adaptive training resources based on sensitivity")
    
    # Ablation study flags
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["mpq_only", "sgra_only", "all"],
                        help="Run specific ablation experiment")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()

    # Set up ablation experiments
    if args.ablation == "mpq_only":
        args.use_mixed_precision = True
        args.use_adaptive_training = False
    elif args.ablation == "sgra_only":
        args.use_mixed_precision = False
        args.use_adaptive_training = True
    elif args.ablation == "all":
        args.use_mixed_precision = True
        args.use_adaptive_training = True
    
    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create output directories
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    
    logger.info("="*70)
    logger.info("RESEARCH EXPERIMENT: Sensitivity-Guided Mixed-Precision Quantization")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Sensitivity file: {args.sensitivity_file}")
    logger.info(f"  MPQ enabled: {args.use_mixed_precision}")
    if args.use_mixed_precision:
        logger.info(f"    Strategy: {args.mpq_strategy}")
        logger.info(f"    Target avg bits: {args.target_avg_bits}")
    logger.info(f"  SGRA enabled: {args.use_adaptive_training}")
    logger.info("="*70)
    logger.info(args)

    if args.net is None:
        args.net = args.model.split('/')[-1]
        logger.info(f"Setting net as {args.net}")
    
    if args.resume_quant:
        model, tokenizer = load_quantized_model(
            args.resume_quant, args.wbits, args.group_size)
        logger.info(
            f"memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB")
    else:
        # Load FP16 model
        config = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=False, legacy=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, device_map='cpu', torch_dtype=torch.float16)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        for param in model.parameters():
            param.requires_grad = False

        # Quantization
        if args.wbits < 16:
            logger.info("=== START RESEARCH QUANTIZATION ===")
            tick = time.time()
            
            # Load calibration dataset
            cache_trainloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_valloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_trainloader)
                torch.save(valloader, cache_valloader)
            
            # Run research quantization
            block_ap(
                model,
                args,
                trainloader,
                valloader,
                logger,
            )
            
            training_time = time.time() - tick
            logger.info(f"[RESULTS] Total training time: {training_time:.2f}s ({training_time/60:.2f}min)")
    
    torch.cuda.empty_cache()
    
    # Save quantized model
    if args.save_quant_dir:
        logger.info("Saving quantized model...")
        model.save_pretrained(args.save_quant_dir)
        tokenizer.save_pretrained(args.save_quant_dir)
        logger.info(f"Model saved to {args.save_quant_dir}")
    
    # Evaluation
    logger.info("=== EVALUATION ===")
    results = evaluate(model, tokenizer, args, logger)
    
    # Save results
    results_file = f"{args.output_dir}/results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'configuration': {
                'model': args.model,
                'use_mixed_precision': args.use_mixed_precision,
                'use_adaptive_training': args.use_adaptive_training,
                'mpq_strategy': args.mpq_strategy if args.use_mixed_precision else None,
                'target_avg_bits': args.target_avg_bits if args.use_mixed_precision else None,
            },
            'results': results,
            'training_time': training_time if 'training_time' in locals() else None,
        }, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("="*70)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    print(sys.argv)
    main()


