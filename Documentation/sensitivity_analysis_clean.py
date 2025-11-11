"""
Cleaned Sensitivity Analysis for LLM Quantization
===================================================

This script computes Fisher Information-based sensitivity scores for transformer layers.
Designed for Google Colab and supports multiple models (Llama-2, Llama-3, Mistral).

Usage:
    python sensitivity_analysis_clean.py --model meta-llama/Llama-2-7b-hf --dataset wikitext --samples 64

Author: Research Project
Date: 2025
"""

import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')


class SensitivityAnalyzer:
    """
    Compute layer-wise sensitivity scores for LLMs using Fisher Information.

    Fisher Information measures the expected squared gradient, capturing how
    sensitive the loss is to changes in layer parameters.
    """

    def __init__(self, model_name, dataset_name="wikitext", dataset_config=None,
                 num_samples=64, max_length=512, device=None):
        """
        Initialize the sensitivity analyzer.

        Args:
            model_name: HuggingFace model identifier
            dataset_name: Dataset name ("wikitext", "c4", "redpajama")
            dataset_config: Dataset configuration (e.g., "wikitext-2-raw-v1")
            num_samples: Number of calibration samples
            max_length: Maximum sequence length (512 recommended for 7B models)
            device: Compute device (auto-detected if None)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.num_samples = num_samples
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Sensitivity Analyzer]")
        print(f"  Model: {model_name}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Samples: {num_samples}")
        print(f"  Max Length: {max_length}")
        print(f"  Device: {self.device}")

        self.model = None
        self.tokenizer = None
        self.calibration_data = None

    def load_model(self):
        """Load model and tokenizer with memory optimization."""
        print(f"\n[Loading Model]")

        # Load with FP16 to save memory
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # Automatic device placement
            trust_remote_code=True  # For models like Mistral
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Disable gradient checkpointing for sensitivity computation
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

        # Set to eval mode
        self.model.eval()

        # Get model info
        num_layers = len(self.model.model.layers)
        print(f"  ✓ Model loaded: {num_layers} layers")
        print(f"  ✓ Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        return self.model, self.tokenizer

    def load_calibration_data(self):
        """Load and prepare calibration dataset."""
        print(f"\n[Loading Dataset: {self.dataset_name}]")

        if self.dataset_name == "wikitext":
            config = self.dataset_config or "wikitext-2-raw-v1"
            dataset = load_dataset("wikitext", config, split="train")
            texts = [text for text in dataset["text"] if len(text) > 100]

        elif self.dataset_name == "c4":
            dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
            texts = []
            for i, sample in enumerate(dataset):
                if len(texts) >= self.num_samples:
                    break
                if len(sample["text"]) > 100:
                    texts.append(sample["text"])

        elif self.dataset_name == "redpajama":
            dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
            texts = [text for text in dataset["text"] if len(text) > 100]

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.calibration_data = texts[:self.num_samples]
        print(f"  ✓ Loaded {len(self.calibration_data)} calibration samples")

        return self.calibration_data

    def compute_fisher_sensitivity(self):
        """
        Compute Fisher Information-based sensitivity for each layer.

        Fisher Information = E[∇²L] ≈ average of squared gradients

        Returns:
            List of sensitivity scores (one per layer)
        """
        print(f"\n[Computing Fisher Sensitivity]")
        print(f"  Method: Fisher Information")
        print(f"  Processing {self.num_samples} samples...")

        # Prepare inputs
        inputs = self.tokenizer(
            self.calibration_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # Get model layers
        layers = self.model.model.layers
        num_layers = len(layers)

        # Store sensitivity scores
        sensitivity_scores = []

        # Compute per layer
        for layer_idx, layer in enumerate(tqdm(layers, desc="Analyzing layers")):
            fisher_sum = 0.0

            for sample_idx in range(self.num_samples):
                # Forward pass (single sample)
                outputs = self.model(
                    input_ids=inputs.input_ids[sample_idx:sample_idx+1],
                    attention_mask=inputs.attention_mask[sample_idx:sample_idx+1]
                )
                logits = outputs.logits

                # Compute cross-entropy loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[sample_idx:sample_idx+1, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Backward pass
                self.model.zero_grad()
                loss.backward(retain_graph=True)

                # Compute Fisher information (sum of squared gradients)
                grad_norm_sq = 0.0
                for param in layer.parameters():
                    if param.grad is not None:
                        # Square of L2 norm of gradients
                        grad_norm_sq += (param.grad ** 2).sum().item()

                fisher_sum += grad_norm_sq

                # Clear memory
                torch.cuda.empty_cache()

            # Average across samples
            avg_fisher = fisher_sum / self.num_samples

            # Check for numerical issues
            if not np.isfinite(avg_fisher):
                print(f"  Warning: Layer {layer_idx} has non-finite Fisher value")
                avg_fisher = 1e6  # Fallback

            sensitivity_scores.append(avg_fisher)

            # Progress update
            if (layer_idx + 1) % 8 == 0:
                print(f"  Processed {layer_idx + 1}/{num_layers} layers")

        print(f"  ✓ Sensitivity computation complete")
        return sensitivity_scores

    def analyze_and_rank(self, sensitivity_scores):
        """Analyze sensitivity patterns and rank layers."""
        scores = np.array(sensitivity_scores)

        # Rank by sensitivity (descending)
        ranked_indices = np.argsort(scores)[::-1].tolist()

        # Statistics
        stats = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "most_sensitive_layer": int(scores.argmax()),
            "least_sensitive_layer": int(scores.argmin()),
            "sensitivity_range": float(scores.max() / scores.min())
        }

        return ranked_indices, stats

    def visualize(self, sensitivity_scores, save_path=None):
        """Create visualization of sensitivity scores."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        num_layers = len(sensitivity_scores)
        layer_indices = range(num_layers)

        # Bar chart
        axes[0].bar(layer_indices, sensitivity_scores, color='skyblue', edgecolor='black')
        axes[0].set_title(f'Layer Sensitivity Scores - {self.model_name.split("/")[-1]}',
                          fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Layer Index', fontsize=12)
        axes[0].set_ylabel('Fisher Information', fontsize=12)
        axes[0].grid(True, axis='y', alpha=0.3)

        # Highlight most sensitive layers
        top_5_indices = np.argsort(sensitivity_scores)[-5:]
        for idx in top_5_indices:
            axes[0].bar(idx, sensitivity_scores[idx], color='red', alpha=0.7)

        # Line plot
        axes[1].plot(layer_indices, sensitivity_scores, 'o-', linewidth=2, markersize=6)
        axes[1].set_title('Sensitivity Trend Across Layers', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Layer Index', fontsize=12)
        axes[1].set_ylabel('Fisher Information', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Annotations
        most_idx = np.argmax(sensitivity_scores)
        least_idx = np.argmin(sensitivity_scores)

        axes[1].annotate(f'Most: Layer {most_idx}\n{sensitivity_scores[most_idx]:.2e}',
                        xy=(most_idx, sensitivity_scores[most_idx]),
                        xytext=(most_idx, sensitivity_scores[most_idx] * 1.15),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, ha='center')

        axes[1].annotate(f'Least: Layer {least_idx}\n{sensitivity_scores[least_idx]:.2e}',
                        xy=(least_idx, sensitivity_scores[least_idx]),
                        xytext=(least_idx, sensitivity_scores[least_idx] * 1.15),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=10, ha='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Visualization saved: {save_path}")

        plt.show()

    def save_results(self, sensitivity_scores, ranked_layers, stats, output_file):
        """Save results to JSON file compatible with QAT code."""

        # Prepare results dictionary
        results = {
            "model": {
                "name": self.model_name,
                "num_layers": len(self.model.model.layers),
                "hidden_size": self.model.config.hidden_size,
                "num_attention_heads": self.model.config.num_attention_heads,
            },
            "dataset": {
                "name": self.dataset_name,
                "config": self.dataset_config,
                "num_samples": self.num_samples,
                "max_length": self.max_length
            },
            "method": "fisher",
            "sensitivity_scores": sensitivity_scores,
            "ranked_layers": ranked_layers,
            "statistics": stats,
            "gpu_info": {
                "device": str(self.device),
                "name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "memory_allocated_gb": torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0,
            }
        }

        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[Results Saved]")
        print(f"  ✓ File: {output_file}")
        print(f"  ✓ Format: Compatible with main_research.py")

        # Print summary
        print(f"\n[Summary Statistics]")
        print(f"  Mean sensitivity: {stats['mean']:.4e}")
        print(f"  Std deviation: {stats['std']:.4e}")
        print(f"  Sensitivity range: {stats['sensitivity_range']:.2f}x")
        print(f"  Most sensitive: Layer {stats['most_sensitive_layer']}")
        print(f"  Least sensitive: Layer {stats['least_sensitive_layer']}")
        print(f"  Top 5 layers: {ranked_layers[:5]}")

        return results

    def run_full_analysis(self, output_file=None, visualize=True):
        """Run complete sensitivity analysis pipeline."""
        print("="*70)
        print("SENSITIVITY ANALYSIS - FULL PIPELINE")
        print("="*70)

        # Step 1: Load model
        self.load_model()

        # Step 2: Load data
        self.load_calibration_data()

        # Step 3: Compute sensitivity
        sensitivity_scores = self.compute_fisher_sensitivity()

        # Step 4: Analyze
        ranked_layers, stats = self.analyze_and_rank(sensitivity_scores)

        # Step 5: Visualize
        if visualize:
            self.visualize(sensitivity_scores)

        # Step 6: Save results
        if output_file is None:
            model_short = self.model_name.split('/')[-1].lower().replace('-', '_')
            output_file = f"sensitivity_results_{model_short}.json"

        results = self.save_results(sensitivity_scores, ranked_layers, stats, output_file)

        print("\n" + "="*70)
        print("✓ SENSITIVITY ANALYSIS COMPLETE")
        print("="*70)

        return results


def main():
    """Command-line interface for sensitivity analysis."""
    parser = argparse.ArgumentParser(
        description="Compute Fisher sensitivity scores for LLM quantization"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., meta-llama/Llama-2-7b-hf)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "c4", "redpajama"],
        help="Calibration dataset"
    )

    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset configuration (e.g., wikitext-2-raw-v1)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Number of calibration samples"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path"
    )

    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Skip visualization"
    )

    args = parser.parse_args()

    # Special handling for wikitext config
    if args.dataset == "wikitext" and args.dataset_config is None:
        args.dataset_config = "wikitext-2-raw-v1"

    # Create analyzer
    analyzer = SensitivityAnalyzer(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        num_samples=args.samples,
        max_length=args.max_length
    )

    # Run analysis
    results = analyzer.run_full_analysis(
        output_file=args.output,
        visualize=not args.no_visualize
    )

    print(f"\n✓ Done! Results saved to: {args.output or 'sensitivity_results_*.json'}")


if __name__ == "__main__":
    main()
