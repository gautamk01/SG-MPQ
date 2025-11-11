"""
Research Implementation: Sensitivity-Guided Mixed-Precision Quantization
Novel contributions for publication:
1. Mixed-Precision Quantization (MPQ) - Layer-specific bit-widths
2. Sensitivity-Guided Resource Allocation (SGRA) - Adaptive training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
import pdb
import gc
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os
import json
import numpy as np

def update_dataset(layer, dataset, dev, attention_mask, position_ids):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                new_data = layer(inps, attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
                dataset.update_data(index,new_data)


def calculate_mixed_precision_config(sensitivity_scores, target_avg_bits=4.0, strategy='adaptive'):
    """
    Novel Contribution 1: Mixed-Precision Quantization (MPQ)
    IMPROVED: Rate-distortion theory-based allocation with greedy optimization

    Allocate different bit-widths per layer based on sensitivity scores
    while maintaining a target average bit-width.

    Args:
        sensitivity_scores: Tensor of sensitivity scores per layer
        target_avg_bits: Target average bit-width (e.g., 4.0)
        strategy: 'adaptive', 'aggressive', or 'conservative'

    Returns:
        bit_widths: List of bit-widths per layer
        group_sizes: List of group sizes per layer
    """
    num_layers = len(sensitivity_scores)

    # Normalize sensitivity scores to [0, 1]
    min_s = sensitivity_scores.min()
    max_s = sensitivity_scores.max()
    norm_scores = (sensitivity_scores - min_s) / (max_s - min_s + 1e-8)

    if strategy == 'adaptive':
        # IMPROVED: Power-law mapping (diminishing returns)
        # Theory: Quantization error ~ 2^(-bits), sensitive layers benefit more from additional bits
        alpha = 0.5  # Concave mapping (sqrt-like)
        bit_scores = norm_scores ** alpha

        # IMPROVED: Quantile-based allocation (more principled than hard thresholds)
        bit_widths = []
        quantile_80 = torch.quantile(bit_scores, 0.80)
        quantile_60 = torch.quantile(bit_scores, 0.60)
        quantile_30 = torch.quantile(bit_scores, 0.30)
        quantile_15 = torch.quantile(bit_scores, 0.15)

        for score in bit_scores:
            if score > quantile_80:  # Top 20% most sensitive
                bits = 8
            elif score > quantile_60:  # 60-80% sensitive
                bits = 6
            elif score > quantile_30:  # 30-60% sensitive
                bits = 4
            elif score > quantile_15:  # 15-30% sensitive
                bits = 3
            else:  # Bottom 15% least sensitive
                bits = 2
            bit_widths.append(bits)

        # IMPROVED: Greedy optimization to meet target budget
        bit_widths = _optimize_bit_budget_greedy(
            bit_widths, sensitivity_scores.cpu().numpy(), target_avg_bits
        )

    elif strategy == 'aggressive':
        # More aggressive compression on low-sensitivity layers
        bit_widths = []
        for score in norm_scores:
            if score > 0.9:
                bits = 8
            elif score > 0.7:
                bits = 6
            elif score > 0.4:
                bits = 4
            else:
                bits = 2  # Aggressive 2-bit for low sensitivity
            bit_widths.append(bits)

    elif strategy == 'conservative':
        # More conservative, keep most layers at higher bits
        bit_widths = []
        for score in norm_scores:
            if score > 0.7:
                bits = 8
            elif score > 0.4:
                bits = 6
            else:
                bits = 4  # Minimum 4-bit
            bit_widths.append(bits)

    # Calculate adaptive group sizes (smaller for sensitive layers)
    group_sizes = []
    for score in norm_scores:
        if score > 0.8:
            group_size = 64  # Smaller groups for better accuracy
        elif score > 0.5:
            group_size = 128
        else:
            group_size = 256  # Larger groups for efficiency
        group_sizes.append(group_size)

    return bit_widths, group_sizes


def _optimize_bit_budget_greedy(initial_bits, sensitivity, target_avg):
    """
    IMPROVED: Greedy algorithm to optimize bit allocation

    Maximizes: sum(sensitivity[i] * bits[i])
    Subject to: mean(bits) == target_avg

    Args:
        initial_bits: Initial bit allocation
        sensitivity: Sensitivity scores
        target_avg: Target average bits

    Returns:
        Optimized bit allocation
    """
    bits = list(initial_bits)
    current_avg = np.mean(bits)

    # Iteratively adjust to meet budget
    max_iterations = len(bits) * 5
    iteration = 0

    while abs(current_avg - target_avg) > 0.05 and iteration < max_iterations:
        if current_avg > target_avg:
            # Reduce bits from least sensitive high-bit layers
            # Metric: sensitivity per bit (efficiency)
            candidates = [(i, bits[i], sensitivity[i] / bits[i])
                         for i in range(len(bits)) if bits[i] > 2]
            if not candidates:
                break
            candidates.sort(key=lambda x: x[2])  # Sort by efficiency (ascending)
            idx = candidates[0][0]
            bits[idx] = max(2, bits[idx] - 1)
        else:
            # Add bits to most sensitive low-bit layers
            # Metric: sensitivity per (bits + 1) (marginal benefit)
            candidates = [(i, bits[i], sensitivity[i] / (bits[i] + 1))
                         for i in range(len(bits)) if bits[i] < 8]
            if not candidates:
                break
            candidates.sort(key=lambda x: -x[2])  # Sort by marginal benefit (descending)
            idx = candidates[0][0]
            bits[idx] = min(8, bits[idx] + 1)

        current_avg = np.mean(bits)
        iteration += 1

    return bits


def calculate_adaptive_training_config(sensitivity_scores, base_epochs=2, base_lr=1e-4):
    """
    Novel Contribution 2: Sensitivity-Guided Resource Allocation (SGRA)
    IMPROVED: Theoretically grounded resource allocation with sqrt scaling

    Theory: PAC learning theory suggests sample complexity scales as sqrt(VC dimension)
    High sensitivity → Higher VC dimension → Need more samples (epochs) + smaller steps (LR)

    Allocate training resources (epochs, LR, patience) based on sensitivity.

    Returns:
        training_configs: List of dicts with per-layer training settings
    """
    num_layers = len(sensitivity_scores)

    # Normalize sensitivity to [0, 1]
    min_s = sensitivity_scores.min()
    max_s = sensitivity_scores.max()
    norm_scores = (sensitivity_scores - min_s) / (max_s - min_s + 1e-8)

    training_configs = []

    for idx, score in enumerate(norm_scores):
        # IMPROVED: Square-root epoch scaling (PAC learning theory)
        # More epochs for sensitive layers, but with diminishing returns
        epoch_multiplier = 1.0 + math.sqrt(score)  # Range: 1.0 (score=0) to 2.0 (score=1)
        adaptive_epochs = int(base_epochs * epoch_multiplier)

        # IMPROVED: Inverse LR scaling (smaller LR for sensitive layers)
        # Theory: Sensitive layers need more conservative updates to avoid overfitting
        # Higher sensitivity → Lower LR (more stability)
        lr_multiplier = 1.0 / (1.0 + 0.5 * score)  # Range: 1.0 (score=0) to 0.67 (score=1)
        adaptive_lr = base_lr * lr_multiplier

        # IMPROVED: Continuous patience scaling (no hard thresholds)
        # More patience for sensitive layers (allow more convergence time)
        patience = int(2 + 3 * score)  # Range: 2 (score=0) to 5 (score=1)

        # IMPROVED: Continuous validation frequency
        # More frequent validation for sensitive layers (tighter monitoring)
        val_freq = 1 if score > 0.6 else 2  # Validate every epoch for top 40%

        # Weight LR: Even smaller for sensitive layers (10% of quant LR)
        adaptive_weight_lr = adaptive_lr * 0.1

        config = {
            'layer_idx': idx,
            'sensitivity': sensitivity_scores[idx].item(),
            'epochs': adaptive_epochs,
            'quant_lr': adaptive_lr,
            'weight_lr': adaptive_weight_lr,
            'patience': patience,
            'val_freq': val_freq,
        }
        training_configs.append(config)

    return training_configs


def joint_mpq_sgra_optimization(sensitivity_scores, target_avg_bits=4.0, base_epochs=2, base_lr=1e-4, strategy='adaptive'):
    """
    NEW: Joint MPQ + SGRA Optimization

    Coordinates bit allocation and training resource allocation for maximum synergy.
    Theory: Layers with higher bits benefit more from training (more capacity to learn),
    so we should align training budget with bit allocation.

    Args:
        sensitivity_scores: Tensor of layer sensitivities
        target_avg_bits: Target average bit-width
        base_epochs: Base number of epochs
        base_lr: Base learning rate
        strategy: MPQ strategy ('adaptive', 'aggressive', 'conservative')

    Returns:
        bit_widths: Optimized bit-widths per layer
        group_sizes: Optimized group sizes per layer
        training_configs: Optimized training configs per layer (aligned with bit allocation)
    """
    # Step 1: Get MPQ configuration
    bit_widths, group_sizes = calculate_mixed_precision_config(
        sensitivity_scores, target_avg_bits=target_avg_bits, strategy=strategy
    )

    # Step 2: Get base SGRA configuration
    training_configs = calculate_adaptive_training_config(
        sensitivity_scores, base_epochs=base_epochs, base_lr=base_lr
    )

    # Step 3: JOINT OPTIMIZATION - Align training budget with bit allocation
    # Theory: Higher bits → More capacity → Need more training to utilize that capacity
    bit_widths_tensor = torch.tensor(bit_widths, dtype=torch.float32)
    bit_norm = (bit_widths_tensor - bit_widths_tensor.min()) / (bit_widths_tensor.max() - bit_widths_tensor.min() + 1e-8)

    for idx, config in enumerate(training_configs):
        bit_factor = bit_norm[idx].item()  # 0 (low bits) to 1 (high bits)

        # SYNERGY 1: Boost epochs for high-bit layers (they need more training to converge)
        # Original epochs already scaled by sensitivity, now add bit-based boost
        epoch_boost = 1.0 + 0.3 * bit_factor  # Up to 30% more epochs for 8-bit layers
        config['epochs'] = int(config['epochs'] * epoch_boost)

        # SYNERGY 2: Adjust LR based on bit-width
        # Higher bits → More precision → Can use slightly higher LR
        lr_boost = 1.0 + 0.2 * bit_factor  # Up to 20% higher LR for 8-bit layers
        config['quant_lr'] = config['quant_lr'] * lr_boost
        config['weight_lr'] = config['weight_lr'] * lr_boost

        # SYNERGY 3: Patience scales with bits (more capacity needs more convergence time)
        patience_boost = int(bit_factor * 2)  # +0 to +2 patience for high bits
        config['patience'] = config['patience'] + patience_boost

        # Add bit-width info to config for tracking
        config['bit_width'] = bit_widths[idx]
        config['group_size'] = group_sizes[idx]

    return bit_widths, group_sizes, training_configs


def block_ap(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
):
    """
    Research Version: Block-wise Adaptive Precision Quantization
    with Sensitivity-Guided Mixed-Precision and Resource Allocation
    """
    logger.info("Starting Research Implementation: Sensitivity-Guided Mixed-Precision QAT")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # step 1: move embedding layer and first layer to target device
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    
    # step 3: catch the input of the first layer 
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    layers[0] = layers[0].module
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info("No attention mask caught from the first layer.")
        attention_mask_batch = None
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input
    if args.off_load_to_disk:
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    # RESEARCH CONTRIBUTION: Load and process sensitivity scores
    sensitivity_scores = None
    bit_widths = None
    group_sizes = None
    training_configs = None
    
    if hasattr(args, 'sensitivity_file') and args.sensitivity_file and os.path.exists(args.sensitivity_file):
        logger.info(f"[RESEARCH] Loading sensitivity scores from {args.sensitivity_file}")
        with open(args.sensitivity_file, 'r') as f:
            sensitivity_data = json.load(f)
        sensitivity_scores = torch.tensor(sensitivity_data['sensitivity_scores'])
        logger.info(f"[RESEARCH] Loaded sensitivity scores for {len(sensitivity_scores)} layers")
        logger.info(f"[RESEARCH] Sensitivity range: {sensitivity_scores.min():.4f} to {sensitivity_scores.max():.4f}")
        
        # Contribution 1: Mixed-Precision Quantization (MPQ)
        if hasattr(args, 'use_mixed_precision') and args.use_mixed_precision:
            strategy = getattr(args, 'mpq_strategy', 'adaptive')
            target_bits = getattr(args, 'target_avg_bits', 4.0)
            bit_widths, group_sizes = calculate_mixed_precision_config(
                sensitivity_scores, target_avg_bits=target_bits, strategy=strategy
            )
            logger.info(f"[MPQ] Mixed-Precision Configuration:")
            logger.info(f"  Strategy: {strategy}, Target Avg: {target_bits} bits")
            logger.info(f"  Bit-widths: {bit_widths}")
            logger.info(f"  Actual Avg: {sum(bit_widths)/len(bit_widths):.2f} bits")
            logger.info(f"  Compression: {16/np.mean(bit_widths):.2f}x vs FP16")
        
        # Contribution 2: Sensitivity-Guided Resource Allocation (SGRA)
        if hasattr(args, 'use_adaptive_training') and args.use_adaptive_training:
            training_configs = calculate_adaptive_training_config(
                sensitivity_scores, base_epochs=args.epochs, base_lr=args.quant_lr
            )
            logger.info(f"[SGRA] Adaptive Training Configuration:")
            logger.info(f"  Epoch range: {min(c['epochs'] for c in training_configs)}-{max(c['epochs'] for c in training_configs)}")
            logger.info(f"  LR range: {min(c['quant_lr'] for c in training_configs):.2e}-{max(c['quant_lr'] for c in training_configs):.2e}")
        

    # Statistics tracking for paper
    layer_stats = []
    
    # step 6: start training    
    loss_func = torch.nn.MSELoss()
    for block_index in range(len(layers)):
        layer_start_time = time.time()
        
        # Get layer-specific configuration
        layer_wbits = bit_widths[block_index] if bit_widths else args.wbits
        layer_group_size = group_sizes[block_index] if group_sizes else args.group_size
        layer_config = training_configs[block_index] if training_configs else None
        layer_sensitivity = sensitivity_scores[block_index].item() if sensitivity_scores is not None else 0.0
        
        logger.info(f"=== Start quantize block {block_index} ===")
        logger.info(f"[CONFIG] Sensitivity: {layer_sensitivity:.4f}, Bits: {layer_wbits}, Group: {layer_group_size}")
        
        if layer_config:
            logger.info(f"[SGRA] Epochs: {layer_config['epochs']}, LR: {layer_config['quant_lr']:.2e}, Patience: {layer_config['patience']}")
        
        # step 6.1: replace torch.nn.Linear with QuantLinear for QAT
        layer = layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module,torch.nn.Linear):
                # RESEARCH: Use layer-specific bit-width and group size
                quantlinear = int_linear_fake.QuantLinear(module, layer_wbits, layer_group_size)
                set_op_by_name(qlayer, name, quantlinear)  
                del module  
        qlayer.to(dev)
        
        # step 6.2: obtain output of full-precision model for MSE
        set_quant_state(qlayer,weight_quant=False)
        adaptive_epochs = layer_config['epochs'] if layer_config else args.epochs
        
        if adaptive_epochs > 0:
            update_dataset(qlayer,fp_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayer,fp_val_inps,dev,attention_mask,position_ids)
        set_quant_state(qlayer,weight_quant=True)
        
        if adaptive_epochs > 0:
            with torch.no_grad():
                qlayer.float()
            
            # step 6.3: create optimizer with adaptive learning rates
            param = []
            param_group_index = 0
            
            # Use layer-specific learning rates if available
            layer_quant_lr = layer_config['quant_lr'] if layer_config else args.quant_lr
            layer_weight_lr = layer_config['weight_lr'] if layer_config else args.weight_lr
            
            total_training_iteration = adaptive_epochs * args.train_size / args.batch_size 
            
            if layer_quant_lr > 0:
                set_quant_parameters(qlayer,True)
                param.append({"params":quant_parameters(qlayer),"lr":layer_quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=layer_quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=layer_quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)
                
            if layer_weight_lr > 0:
                set_weight_parameters(qlayer,True)
                param.append({"params":weight_parameters(qlayer),"lr":layer_weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=layer_weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=layer_weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer,False)
            
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            
            best_val_loss = 1e6
            early_stop_flag = 0
            layer_patience = layer_config['patience'] if layer_config else args.early_stop
            
            epoch_losses = []
            
            for epoch in range(adaptive_epochs):
                loss_list = []
                norm_list = []
                start_time = time.time()
                
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):    
                    with torch.cuda.amp.autocast():
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        reconstruction_loss = loss_func(label, quant_out)
                        loss =  reconstruction_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
                    norm_list.append(norm.data)

                    if layer_quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                    if layer_weight_lr >0 :
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]

                # Validation
                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):  
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())
                 
                train_mean_num = min(len(loss_list),64)
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                
                epoch_losses.append(val_loss_mean.item())
                
                logger.info(f"[TRAINING] Block {block_index} Epoch {epoch}/{adaptive_epochs} | Train Loss: {loss_mean:.6f} | Val Loss: {val_loss_mean:.6f} | LR: {quant_scheduler.get_lr()[0]:.2e} | Time: {time.time()-start_time:.1f}s")
                
                # Adaptive early stopping
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                    early_stop_flag = 0
                else:
                    early_stop_flag += 1
                    if layer_patience > 0 and early_stop_flag >= layer_patience:
                        logger.info(f"[EARLY STOP] Block {block_index} stopped at epoch {epoch}")
                        break
            
            optimizer.zero_grad()
            del optimizer
            
            # Collect statistics for paper
            layer_stats.append({
                'layer_idx': block_index,
                'sensitivity': layer_sensitivity,
                'bit_width': layer_wbits,
                'group_size': layer_group_size,
                'epochs_trained': epoch + 1,
                'final_val_loss': best_val_loss.item() if torch.is_tensor(best_val_loss) else best_val_loss,
                'training_time': time.time() - layer_start_time,
            })

        # step 6.6: quantize weights inplace
        qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer,weight_quant=False)

        # step 6.7: update inputs
        if adaptive_epochs>0:
            update_dataset(qlayer,quant_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayer,quant_val_inps,dev,attention_mask,position_ids)
        layers[block_index] = qlayer.to("cpu")

        # step 7: pack quantized weights
        if args.real_quant:
            named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(layer_wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)       
                del module        
        del layer
        torch.cuda.empty_cache()

    # Save research statistics
    if hasattr(args, 'output_dir') and layer_stats:
        stats_file = f"{args.output_dir}/layer_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump({
                'layer_stats': layer_stats,
                'average_bits': sum(s['bit_width'] for s in layer_stats) / len(layer_stats),
                'total_training_time': sum(s['training_time'] for s in layer_stats),
                'config': {
                    'use_mixed_precision': getattr(args, 'use_mixed_precision', False),
                    'use_adaptive_training': getattr(args, 'use_adaptive_training', False),
                    'mpq_strategy': getattr(args, 'mpq_strategy', 'adaptive'),
                }
            }, f, indent=2)
        logger.info(f"[RESEARCH] Statistics saved to {stats_file}")

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

