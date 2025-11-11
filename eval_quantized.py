"""
Standalone Evaluation Script for Quantized Models

Usage:
    # Evaluate a saved mixed-precision model
    python eval_quantized.py --model_path ./output/mpq_adaptive/model \
        --eval_ppl --eval_tasks piqa,arc_easy,hellaswag
    
    # Evaluate a uniform quantized model
    python eval_quantized.py --model_path ./saved_model \
        --wbits 4 --group_size 128 \
        --eval_ppl --eval_tasks piqa,arc_easy
"""

import os
import sys
import argparse
import torch
import json
from transformers import AutoTokenizer
from quantize.int_linear_real import load_quantized_model, load_mixed_precision_quantized_model
from accelerate import infer_auto_device_map, dispatch_model
from datautils_block import test_ppl


@torch.no_grad()
def evaluate(model, tokenizer, args):
    '''Evaluate model performance'''
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(
        model, 
        max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
        no_split_module_classes=[block_class_name]
    )
    model = dispatch_model(model, device_map=device_map)
    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            print(f'[RESULTS] {dataset} perplexity: {ppl_results[dataset]:.2f}')
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
        print(make_table(eval_results))
        total_acc = 0
        for task in task_list:
            total_acc += eval_results['results'][task]['acc,none']
            results[f'{task}_acc'] = eval_results['results'][task]['acc,none']
        avg_acc = total_acc/len(task_list)*100
        print(f'[RESULTS] Average Acc: {avg_acc:.2f}%')
        results['avg_acc'] = avg_acc
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quantized Models")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved quantized model")
    parser.add_argument("--wbits", type=int, default=4,
                        help="Bit-width for uniform quantization (ignored for mixed-precision)")
    parser.add_argument("--group_size", type=int, default=128,
                        help="Group size for uniform quantization (ignored for mixed-precision)")
    
    # Evaluation arguments
    parser.add_argument("--eval_ppl", action="store_true",
                        help="Evaluate perplexity on wikitext2")
    parser.add_argument("--ppl_seqlen", type=int, default=2048,
                        help="Sequence length for perplexity evaluation")
    parser.add_argument("--eval_tasks", type=str, default="",
                        help="Comma-separated list of tasks (e.g., piqa,arc_easy,hellaswag)")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Batch size for task evaluation")
    
    # Device arguments
    parser.add_argument("--max_memory", type=str, default="70GiB",
                        help="Maximum memory per GPU")
    
    # Output
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional: Save results to JSON file")
    
    args = parser.parse_args()
    
    print("="*70)
    print("QUANTIZED MODEL EVALUATION")
    print("="*70)
    print(f"Model path: {args.model_path}")
    
    # Detect if this is a mixed-precision model
    is_mixed_precision = False
    # Try model directory first, then parent directory
    stats_file = os.path.join(args.model_path, "layer_statistics.json")
    if not os.path.exists(stats_file):
        # Check parent directory (common structure: output_dir/model/)
        parent_stats_file = os.path.join(os.path.dirname(args.model_path), "layer_statistics.json")
        if os.path.exists(parent_stats_file):
            stats_file = parent_stats_file
    
    if os.path.exists(stats_file):
        print(f"Found layer statistics: {stats_file}")
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            if 'layer_stats' in stats and len(stats['layer_stats']) > 0:
                bit_widths = [s['bit_width'] for s in stats['layer_stats']]
                if len(set(bit_widths)) > 1:
                    is_mixed_precision = True
                    print(f"Detected mixed-precision model")
                    print(f"  Bit-widths used: {sorted(set(bit_widths))}")
                    print(f"  Average bits: {sum(bit_widths)/len(bit_widths):.2f}")
                    
                    # Show distribution
                    from collections import Counter
                    bit_dist = Counter(bit_widths)
                    print("  Distribution:")
                    for bits in sorted(bit_dist.keys()):
                        count = bit_dist[bits]
                        percentage = (count / len(bit_widths)) * 100
                        print(f"    {bits}-bit: {count} layers ({percentage:.1f}%)")
    
    # Load model
    if is_mixed_precision:
        print("\nLoading mixed-precision quantized model...")
        model, tokenizer = load_mixed_precision_quantized_model(args.model_path)
    else:
        print(f"\nLoading uniform quantized model ({args.wbits}-bit, group_size={args.group_size})...")
        model, tokenizer = load_quantized_model(args.model_path, args.wbits, args.group_size)
    
    print("Model loaded successfully!")
    print("="*70)
    
    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate(model, tokenizer, args)
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    for key, value in results.items():
        if 'ppl' in key:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value:.4f}")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump({
                'model_path': args.model_path,
                'is_mixed_precision': is_mixed_precision,
                'results': results,
            }, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
    
    print("="*70)


if __name__ == "__main__":
    main()
