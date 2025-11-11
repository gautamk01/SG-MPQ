## 📋 Project Summary

This project implements **Sensitivity-Guided Mixed-Precision Quantization-Aware Training (SG-MPQ)** for Large Language Models. The approach combines three novel contributions:

1. **Mixed-Precision Quantization (MPQ)** - Rate-distortion optimized bit allocation
2. **Sensitivity-Guided Resource Allocation (SGRA)** - Theoretically grounded training adaptation
3. **Joint MPQ+SGRA Optimization** - Coordinated bit and training budget allocation

---

## 🎯 Research Contributions

### 1. Mixed-Precision Quantization (MPQ) - IMPROVED

**Previous Approach:**
- Linear sensitivity mapping to bit-widths
- Hard-coded thresholds (0.8, 0.6, etc.)
- Sequential bit adjustment

**Our Improvements:**
- **Power-law mapping** (α=0.5): Captures diminishing returns of additional bits
- **Quantile-based allocation**: Dynamic thresholds adapt to sensitivity distribution
- **Greedy optimization**: Maximizes `Σ(sensitivity[i] × bits[i])` subject to budget constraint

**Expected Impact:** ~0.5-1.0 PPL improvement vs baseline MPQ

### 2. Sensitivity-Guided Resource Allocation (SGRA) - IMPROVED

**Previous Approach:**
- Linear epoch scaling: `epochs = base × (1 + sensitivity)`
- Linear LR scaling
- Hard-coded patience thresholds

**Our Improvements:**
- **Square-root epoch scaling**: `epochs = base × (1 + √sensitivity)` (PAC learning theory)
- **Inverse LR scaling**: `lr = base / (1 + 0.5 × sensitivity)` (stability for sensitive layers)
- **Continuous patience**: `patience = 2 + 3 × sensitivity` (no thresholds)

**Theoretical Grounding:** PAC learning theory - sample complexity scales as √(VC dimension)

**Expected Impact:** ~15-20% faster training, ~0.5 PPL improvement

### 3. Joint MPQ+SGRA Optimization - NEW

**Key Insight:** Layers with higher bits have more capacity and benefit from more training

**Synergies:**
1. **Epoch boost**: High-bit layers get +30% more epochs
2. **LR boost**: High-bit layers can use +20% higher LR (more precision)
3. **Patience boost**: High-bit layers get +2 extra patience (more convergence time)

**Expected Impact:** ~0.3-0.5 PPL improvement vs separate MPQ+SGRA


## 🔬 Experimental Setup

### Models
- Llama-2-7B (baseline)
- Llama-3-8B (validation)
- Mistral-7B (validation)

### Datasets
- **Calibration + QAT:** WikiText-2 (MUST match for both!)
- **Evaluation:** WikiText-2 (PPL), PIQA, ARC-Easy, HellaSwag (accuracy)

### Resource Constraints
- **GPU:** L4 24GB
- **Training Size:** 128-512 samples
- **Validation Size:** 16 samples

### Experiments
1. **Baseline:** Uniform 4-bit (no sensitivity)
2. **MPQ Only:** Improved mixed-precision
3. **SGRA Only:** Improved adaptive training
4. **MPQ+SGRA Combined:** Separate optimization
5. **Joint MPQ+SGRA:** Coordinated optimization (NEW)


