"""
PromptSage Comprehensive Research and Evaluation Script

This script performs a complete research evaluation of PromptSage:
1. Downloads and prepares datasets from Hugging Face
2. Trains the meta-learning optimizer
3. Evaluates multiple strategies with 1B parameter models
4. Generates publication-quality figures for a research paper
"""

import os
import json
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Any

# Import transformers and datasets
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GPTNeoForCausalLM,
    set_seed
)

# Import PromptSage components
from promptsage import PromptOptimizer
from promptsage.core.analyzer import PromptAnalyzer
from promptsage.core.generator import PromptGenerator
from promptsage.core.evaluator import PromptEvaluator
from promptsage.models.meta_learner import MetaLearningOptimizer
from promptsage.models.evolution import EvolutionaryOptimizer
from promptsage.models.contrastive import ContrastiveOptimizer
from promptsage.utils.visualization import plot_comparison, plot_optimization_history

# NLTK requirements for analyzer
import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Set random seed for reproducibility
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Create custom colormap for visualizations
promptsage_colors = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]
promptsage_cmap = LinearSegmentedColormap.from_list("promptsage", promptsage_colors)

# Configure Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

# Configure argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="PromptSage Research Evaluation")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-1.3B", 
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="research_results", 
                        help="Output directory for results and figures")
    parser.add_argument("--gpu", type=str, default="0", 
                        help="GPU device index to use")
    
    # Training parameters
    parser.add_argument("--meta_epochs", type=int, default=3, 
                        help="Number of meta-training epochs")
    parser.add_argument("--train_samples", type=int, default=500, 
                        help="Number of samples for training")
    parser.add_argument("--eval_samples", type=int, default=100, 
                        help="Number of samples for evaluation")
    parser.add_argument("--iterations", type=int, default=5, 
                        help="Number of optimization iterations")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="databricks/dolly-15k", 
                        choices=["databricks/dolly-15k", "tatsu-lab/alpaca", "Anthropic/hh-rlhf"],
                        help="Dataset to use")
    
    # Experiment settings
    parser.add_argument("--quick_mode", action="store_true",
                        help="Run in quick mode with smaller models and datasets")
    
    return parser.parse_args()

def setup_environment(args):
    """Set up environment for GPU training."""
    # Set up CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save run configuration
    config = vars(args)
    config["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["device"] = str(device)
    config["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        config["cuda_version"] = torch.version.cuda
        config["gpu_name"] = torch.cuda.get_device_name(device)
        config["gpu_memory"] = f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
    
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return device

def prepare_datasets(args, device):
    """Prepare datasets for training and evaluation."""
    print("\n=== Preparing Datasets ===")
    
    # Load selected dataset
    dataset_name = args.dataset
    print(f"Loading dataset: {dataset_name}")
    
    # Different handling based on dataset structure
    if dataset_name == "databricks/dolly-15k":
        dataset = load_dataset(dataset_name)
        
        # Convert to train/eval splits
        training_data = []
        evaluation_data = []
        
        # Define task type mapping 
        category_to_task = {
            "creative_writing": "creative",
            "brainstorming": "creative",
            "classification": "analysis",
            "closed_qa": "factual",
            "open_qa": "explanation",
            "summarization": "summarization",
            "information_extraction": "analysis",
            "other": "general"
        }
        
        # Process samples with progress bar
        for item in tqdm(dataset["train"], desc="Processing Dolly dataset"):
            # Get category and map to task type
            category = item["category"]
            task_type = category_to_task.get(category, "general")
            
            # Create original prompt from instruction and context
            instruction = item["instruction"]
            context = item["context"]
            response = item["response"]
            
            if context:
                original_prompt = f"{instruction}\n\nContext: {context}"
            else:
                original_prompt = instruction
            
            # Create entry
            entry = {
                "original_prompt": original_prompt,
                "task_type": task_type,
                "reference": response
            }
            
            # Randomly assign to training or evaluation
            if random.random() < 0.8:  # 80% for training
                training_data.append(entry)
            else:  # 20% for evaluation
                evaluation_data.append(entry)
                
    elif dataset_name == "tatsu-lab/alpaca":
        dataset = load_dataset("tatsu-lab/alpaca")
        
        # Convert to train/eval splits
        training_data = []
        evaluation_data = []
        
        # Process samples with progress bar
        for item in tqdm(dataset["train"], desc="Processing Alpaca dataset"):
            instruction = item["instruction"]
            input_text = item["input"]
            output = item["output"]
            
            # Create original prompt
            if input_text:
                original_prompt = f"{instruction}\n\n{input_text}"
            else:
                original_prompt = instruction
            
            # Infer task type based on instruction content
            task_type = "general"
            task_keywords = {
                "explain": "explanation",
                "describe": "explanation",
                "write": "creative",
                "create": "creative",
                "code": "coding",
                "program": "coding",
                "compare": "comparison",
                "difference": "comparison",
                "summarize": "summarization",
                "list": "factual",
                "what is": "factual"
            }
            
            for keyword, task in task_keywords.items():
                if keyword in instruction.lower():
                    task_type = task
                    break
            
            # Create entry
            entry = {
                "original_prompt": original_prompt,
                "task_type": task_type,
                "reference": output
            }
            
            # Randomly assign to training or evaluation
            if random.random() < 0.8:  # 80% for training
                training_data.append(entry)
            else:  # 20% for evaluation
                evaluation_data.append(entry)
                
    elif dataset_name == "Anthropic/hh-rlhf":
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        
        # Convert to train/eval splits
        training_data = []
        evaluation_data = []
        
        # Process samples with progress bar
        for item in tqdm(dataset, desc="Processing HH-RLHF dataset"):
            # Extract human query from chosen or rejected response
            if "<human>:" in item["chosen"]:
                parts = item["chosen"].split("<human>:", 1)
                if len(parts) > 1:
                    query = parts[1].split("<assistant>:", 1)[0].strip()
                    helpful_response = item["chosen"].split("<assistant>:", 1)[1].strip()
                    
                    # Create entry
                    entry = {
                        "original_prompt": query,
                        "task_type": "general",  # Generic task type
                        "reference": helpful_response
                    }
                    
                    # Randomly assign to training or evaluation
                    if random.random() < 0.8:  # 80% for training
                        training_data.append(entry)
                    else:  # 20% for evaluation
                        evaluation_data.append(entry)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Limit number of samples if specified
    if args.train_samples > 0 and len(training_data) > args.train_samples:
        random.shuffle(training_data)
        training_data = training_data[:args.train_samples]
    
    if args.eval_samples > 0 and len(evaluation_data) > args.eval_samples:
        random.shuffle(evaluation_data)
        evaluation_data = evaluation_data[:args.eval_samples]
    
    print(f"Prepared {len(training_data)} training samples and {len(evaluation_data)} evaluation samples")
    
    # Generate synthetic optimized prompts for training
    training_data = generate_synthetic_optimizations(training_data, args, device)
    
    # Save datasets
    train_path = os.path.join(args.output_dir, "training_data.json")
    eval_path = os.path.join(args.output_dir, "evaluation_data.json")
    
    with open(train_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    with open(eval_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    return training_data, evaluation_data

def generate_synthetic_optimizations(training_data, args, device):
    """Generate synthetic optimized prompts for training examples."""
    print("\n=== Generating Synthetic Optimized Prompts ===")
    
    # Create prompt analyzer
    analyzer = PromptAnalyzer()
    
    # Load small GPT-2 model for quick generation if needed
    if args.quick_mode:
        model_name = "distilgpt2"
        print(f"Loading {model_name} for synthetic prompt generation...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Initialize generator for creating optimized prompts
        generator = PromptGenerator(model, tokenizer, device=device)
    
    # Process samples with progress bar
    enhanced_training_data = []
    
    for item in tqdm(training_data, desc="Generating optimized prompts"):
        original_prompt = item["original_prompt"]
        task_type = item["task_type"]
        
        # Create optimized versions - strategy 1: Add specificity
        analysis = analyzer.analyze(original_prompt)
        
        # Different optimization techniques
        specificity_additions = [
            "Be specific and include concrete examples.",
            "Please provide a detailed explanation with examples.",
            "Include key points and supporting evidence.",
            "Use precise terminology and clear definitions.",
            "Break down your answer into logical steps.",
            "Explain this as if to a beginner in the field.",
            "Compare different perspectives on this topic.",
            "Include both advantages and disadvantages.",
            "Address common misconceptions about this topic.",
            "Provide historical context where relevant."
        ]
        
        # Create optimized version
        if '?' in original_prompt:
            # For questions, insert before the question mark
            parts = original_prompt.split('?', 1)
            optimized_prompt = f"{parts[0]}? {random.choice(specificity_additions)}{parts[1] if len(parts) > 1 else ''}"
        else:
            # For statements, add to the end
            optimized_prompt = f"{original_prompt}. {random.choice(specificity_additions)}"
            
        # Create second optimization based on task type
        task_specific_additions = {
            "explanation": ["Break this down step-by-step.", "Use analogies to explain complex concepts."],
            "creative": ["Be imaginative and use vivid descriptions.", "Create an engaging narrative."],
            "coding": ["Include comments explaining the code.", "Consider edge cases and optimizations."],
            "comparison": ["Use a structured compare and contrast approach.", "Highlight key similarities and differences."],
            "factual": ["Cite relevant facts and figures.", "Base your response on verified information."],
            "analysis": ["Consider multiple perspectives.", "Evaluate the strengths and weaknesses."],
            "summarization": ["Focus on the most important points.", "Be concise while covering key information."],
            "general": ["Provide a comprehensive response.", "Consider different aspects of the question."]
        }
        
        # Add task-specific enhancement
        additions = task_specific_additions.get(task_type, task_specific_additions["general"])
        task_optimized_prompt = f"{original_prompt}. {random.choice(additions)}"
        
        # Use generator for third optimization if available
        if args.quick_mode and "generator" in locals():
            try:
                # Use meta-prompt to generate an optimized version
                meta_prompt = f"""
                Improve this prompt to get better, more detailed responses:
                
                Original prompt: "{original_prompt}"
                
                Improved prompt:
                """
                
                generated_response = generator.generate(meta_prompt, max_length=len(original_prompt) * 2 + 50)
                
                # Extract generated optimized prompt
                if "Improved prompt:" in generated_response:
                    gen_optimized_prompt = generated_response.split("Improved prompt:", 1)[1].strip()
                    # Clean up quotes if present
                    if gen_optimized_prompt.startswith('"') and gen_optimized_prompt.endswith('"'):
                        gen_optimized_prompt = gen_optimized_prompt[1:-1]
                else:
                    gen_optimized_prompt = generated_response
                
                # Add if it's different enough
                if gen_optimized_prompt != original_prompt and len(gen_optimized_prompt) > len(original_prompt) * 0.5:
                    # Add this as third optimization
                    enhanced_training_data.append({
                        "original_prompt": original_prompt,
                        "optimized_prompt": gen_optimized_prompt,
                        "task_type": task_type,
                        "optimization_method": "generator",
                        "reference": item.get("reference", "")
                    })
            except Exception as e:
                print(f"Error in generator optimization: {e}")
        
        # Add the two rule-based optimizations
        enhanced_training_data.append({
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "task_type": task_type,
            "optimization_method": "specificity",
            "reference": item.get("reference", "")
        })
        
        enhanced_training_data.append({
            "original_prompt": original_prompt,
            "optimized_prompt": task_optimized_prompt,
            "task_type": task_type,
            "optimization_method": "task_specific",
            "reference": item.get("reference", "")
        })
    
    print(f"Generated {len(enhanced_training_data)} optimized prompts for {len(training_data)} original prompts")
    return enhanced_training_data

def load_model(model_name, device, quick_mode=False):
    """Load language model for evaluation."""
    print(f"\n=== Loading Model: {model_name} ===")
    
    # Use smaller model in quick mode
    if quick_mode:
        model_name = "distilgpt2"
        print(f"Quick mode enabled: using {model_name} instead")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure model loading with the right settings for A100
        if "gpt-neo" in model_name.lower() or "pythia" in model_name.lower():
            # Optimize loading for Neo/Pythia models
            model = GPTNeoForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True,
            ).to(device)
        else:
            # General loading for other models
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True,
            ).to(device)
        
        print(f"Model loaded successfully on {device}")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        if not quick_mode:
            print("Falling back to distilgpt2...")
            return load_model("distilgpt2", device, quick_mode=True)
        else:
            raise

def train_meta_optimizer(training_data, model, tokenizer, args, device):
    """Train the meta-learning optimizer on prepared data."""
    print("\n=== Training Meta-Learning Optimizer ===")
    
    # Initialize meta-optimizer
    meta_optimizer = MetaLearningOptimizer()
    
    # Initialize analyzer, generator, and evaluator
    analyzer = PromptAnalyzer()
    generator = PromptGenerator(model, tokenizer, device=device)
    evaluator = PromptEvaluator(model, tokenizer, device=device)
    
    # Training loop
    print(f"Training on {len(training_data)} prompt examples for {args.meta_epochs} epochs...")
    
    # Create contrastive pairs from training data
    contrastive_pairs = []
    
    # For monitoring learning progress
    training_metrics = {
        "epoch": [],
        "improvements": [],
        "pattern_weights": [],
        "task_performance": {}
    }
    
    # Dictionary to track task-specific performance
    task_metrics = {}
    
    for epoch in range(args.meta_epochs):
        print(f"\nEpoch {epoch+1}/{args.meta_epochs}")
        
        # Shuffle data for each epoch
        random.shuffle(training_data)
        
        # Process training examples with progress bar
        all_improvements = []
        epoch_metrics = {"task_improvements": {}}
        
        for item in tqdm(training_data, desc=f"Epoch {epoch+1} training"):
            # Extract data
            original_prompt = item["original_prompt"]
            optimized_prompt = item["optimized_prompt"]
            task_type = item.get("task_type", "general")
            
            # Skip if missing required fields
            if not original_prompt or not optimized_prompt:
                continue
            
            # Initialize task in metrics if not present
            if task_type not in epoch_metrics["task_improvements"]:
                epoch_metrics["task_improvements"][task_type] = []
            
            try:
                # Generate responses
                original_response = generator.generate(original_prompt)
                optimized_response = generator.generate(optimized_prompt)
                
                # Evaluate responses
                original_metrics = evaluator.evaluate(original_prompt, original_response, task_type=task_type)
                optimized_metrics = evaluator.evaluate(optimized_prompt, optimized_response, task_type=task_type)
                
                # Calculate overall improvement
                metric_improvements = []
                
                for metric_name in original_metrics:
                    if metric_name in optimized_metrics:
                        # For perplexity, lower is better
                        if metric_name == "perplexity":
                            if original_metrics[metric_name] > 0:
                                rel_improvement = (original_metrics[metric_name] - optimized_metrics[metric_name]) / original_metrics[metric_name]
                                metric_improvements.append(max(0, rel_improvement)) 
                        else:
                            # For other metrics, higher is better
                            improvement = optimized_metrics[metric_name] - original_metrics[metric_name]
                            metric_improvements.append(max(0, improvement))
                
                if metric_improvements:
                    avg_improvement = sum(metric_improvements) / len(metric_improvements)
                    
                    # Only use examples with actual improvement
                    if avg_improvement > 0:
                        # Add to contrastive pairs for meta-learner
                        contrastive_pairs.append((
                            optimized_prompt, original_prompt, avg_improvement
                        ))
                        
                        # Track improvements
                        all_improvements.append(avg_improvement)
                        epoch_metrics["task_improvements"][task_type].append(avg_improvement)
                        
                        # Update meta-optimizer transformation memory
                        meta_optimizer._update_transformation_memory(
                            task_type, item.get("optimization_method", "general"), avg_improvement
                        )
                        
            except Exception as e:
                print(f"\nError processing example: {e}")
                continue
        
        # Calculate epoch statistics
        if all_improvements:
            avg_improvement = sum(all_improvements) / len(all_improvements)
            print(f"Epoch {epoch+1}: {len(all_improvements)} successful optimizations, Avg improvement: {avg_improvement:.4f}")
        else:
            avg_improvement = 0
            print(f"Epoch {epoch+1}: No successful optimizations")
        
        # Record task-specific performance
        task_summary = {}
        for task, improvements in epoch_metrics["task_improvements"].items():
            if improvements:
                avg_task_improvement = sum(improvements) / len(improvements)
                task_summary[task] = {
                    "count": len(improvements),
                    "avg_improvement": avg_task_improvement
                }
                print(f"  Task '{task}': {len(improvements)} examples, Avg improvement: {avg_task_improvement:.4f}")
        
        # Record training metrics
        training_metrics["epoch"].append(epoch + 1)
        training_metrics["improvements"].append(avg_improvement)
        
        # Record pattern weights
        pattern_weights = {k: float(v) for k, v in meta_optimizer.pattern_weights.items()}
        training_metrics["pattern_weights"].append(pattern_weights)
        
        # Record task performance
        training_metrics["task_performance"][epoch + 1] = task_summary
        
        # Learn from collected contrastive pairs
        if contrastive_pairs:
            meta_optimizer.contrastive_pairs = contrastive_pairs
            meta_optimizer._learn_from_contrastive_pairs()
            print(f"Learned from {len(contrastive_pairs)} contrastive pairs")
            
            # Show top transformations by weight
            sorted_weights = sorted(
                meta_optimizer.pattern_weights.items(), 
                key=lambda x: x[1],
                reverse=True
            )
            print("Top patterns by weight:")
            for pattern, weight in sorted_weights[:5]:
                print(f"  {pattern}: {weight:.4f}")
    
    # Save the trained meta-optimizer
    meta_model_path = os.path.join(args.output_dir, "meta_learner.json")
    meta_optimizer.save_model(meta_model_path)
    print(f"Meta-learning optimizer saved to {meta_model_path}")
    
    # Save training metrics
    metrics_path = os.path.join(args.output_dir, "meta_training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Generate training visualization
    visualize_meta_training(training_metrics, args.output_dir)
    
    return meta_optimizer, meta_model_path, training_metrics

def visualize_meta_training(training_metrics, output_dir):
    """Generate visualizations for meta-learning training progress."""
    print("\n=== Generating Meta-Learning Training Visualizations ===")
    
    # 1. Training progress plot: improvements over epochs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot average improvement
    epochs = training_metrics["epoch"]
    improvements = training_metrics["improvements"]
    
    ax1.plot(epochs, improvements, marker='o', markersize=8, linewidth=2, color=promptsage_colors[0])
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Average Improvement")
    ax1.set_title("Meta-Learning Training Progress", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate values
    for i, (x, y) in enumerate(zip(epochs, improvements)):
        ax1.annotate(f"{y:.4f}", (x, y), xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot pattern weight evolution
    if training_metrics["pattern_weights"]:
        # Get all patterns
        all_patterns = set()
        for weights in training_metrics["pattern_weights"]:
            all_patterns.update(weights.keys())
        
        # Get top 5 patterns by final weight
        if training_metrics["pattern_weights"][-1]:
            final_weights = training_metrics["pattern_weights"][-1]
            top_patterns = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            top_pattern_names = [p[0] for p in top_patterns]
            
            # Extract weights over epochs only for top patterns
            for i, pattern in enumerate(top_pattern_names):
                weights = [epoch_weights.get(pattern, 0) for epoch_weights in training_metrics["pattern_weights"]]
                ax2.plot(epochs, weights, marker='o', linewidth=2, markersize=8, 
                       label=pattern, color=promptsage_colors[i % len(promptsage_colors)])
            
            ax2.set_xlabel("Training Epoch")
            ax2.set_ylabel("Pattern Weight")
            ax2.set_title("Evolution of Top Pattern Weights", fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "meta_training_progress.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Task-specific improvement visualization
    if "task_performance" in training_metrics:
        # Extract task performance data
        task_data = {}
        
        for epoch, epoch_data in training_metrics["task_performance"].items():
            for task, metrics in epoch_data.items():
                if task not in task_data:
                    task_data[task] = {"epochs": [], "improvements": []}
                task_data[task]["epochs"].append(int(epoch))
                task_data[task]["improvements"].append(metrics["avg_improvement"])
        
        if task_data:
            # Create line chart for each task
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, (task, data) in enumerate(task_data.items()):
                if data["epochs"] and data["improvements"]:
                    ax.plot(data["epochs"], data["improvements"], marker='o', linewidth=2, markersize=8,
                          label=f"Task: {task}", color=promptsage_colors[i % len(promptsage_colors)])
            
            ax.set_xlabel("Training Epoch")
            ax.set_ylabel("Average Improvement")
            ax.set_title("Task-Specific Performance Across Training", fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            fig_path = os.path.join(output_dir, "task_training_progress.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create bar chart for final performance by task
            final_improvements = {}
            for task, data in task_data.items():
                if data["improvements"]:
                    final_improvements[task] = data["improvements"][-1]
            
            if final_improvements:
                tasks = list(final_improvements.keys())
                values = list(final_improvements.values())
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(tasks, values, color=promptsage_colors)
                
                ax.set_xlabel("Task Type")
                ax.set_ylabel("Final Average Improvement")
                ax.set_title("Final Meta-Learning Performance by Task", fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f"{value:.4f}", ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                fig_path = os.path.join(output_dir, "final_task_performance.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    print(f"Meta-learning training visualizations saved to {output_dir}")

def evaluate_optimization_strategies(evaluation_data, model, tokenizer, meta_model_path, args, device):
    """Evaluate different optimization strategies on evaluation data."""
    print("\n=== Evaluating Optimization Strategies ===")
    
    # Determine strategies to evaluate
    strategies = ["evolution", "meta", "contrastive"]
    
    # Initialize results storage
    results = {strategy: [] for strategy in strategies}
    
    # Prepare default optimizer
    shared_optimizer = PromptOptimizer(
        model_name=None,  # We'll pass model and tokenizer directly
        optimization_strategy="auto"  # Will be overridden
    )
    
    # Assign model and tokenizer
    shared_optimizer.model = model
    shared_optimizer.tokenizer = tokenizer
    shared_optimizer.device = device
    
    # Initialize components
    shared_optimizer.analyzer = PromptAnalyzer()
    shared_optimizer.generator = PromptGenerator(model, tokenizer, device=device)
    shared_optimizer.evaluator = PromptEvaluator(model, tokenizer, device=device)
    
    # If meta model path is provided, load it
    if meta_model_path:
        print(f"Loading trained meta-learner from {meta_model_path}")
        meta_learner = MetaLearningOptimizer(model_path=meta_model_path)
        shared_optimizer.optimizers["meta"] = meta_learner
    
    # Dictionary to store optimization times
    optimization_times = {strategy: [] for strategy in strategies}
    
    # Baseline responses for comparison
    baseline_responses = {}
    
    print(f"Evaluating {len(evaluation_data)} prompts across {len(strategies)} strategies...")
    
    # First, generate baseline responses for all evaluation prompts
    for i, item in enumerate(tqdm(evaluation_data, desc="Generating baseline responses")):
        prompt = item["original_prompt"]
        
        try:
            # Generate baseline response
            baseline_response = shared_optimizer.generator.generate(prompt, max_length=100)
            baseline_responses[prompt] = baseline_response
        except Exception as e:
            print(f"\nError generating baseline response for prompt {i}: {e}")
    
    # Evaluate each strategy
    for strategy in strategies:
        print(f"\nEvaluating strategy: {strategy}")
        
        # Update optimizer strategy
        shared_optimizer.strategy = strategy
        
        # Process each prompt
        for i, item in enumerate(tqdm(evaluation_data, desc=f"Strategy: {strategy}")):
            prompt = item["original_prompt"]
            task_type = item.get("task_type", "general")
            reference = item.get("reference", None)
            
            # Skip if no baseline response
            if prompt not in baseline_responses:
                continue
            
            baseline_response = baseline_responses[prompt]
            
            try:
                # Time the optimization
                start_time = time.time()
                
                # Run optimization
                optimized_prompt, metrics = shared_optimizer.optimize(
                    prompt=prompt,
                    task_type=task_type,
                    num_iterations=args.iterations,
                    verbose=False
                )
                
                # Record optimization time
                opt_time = time.time() - start_time
                optimization_times[strategy].append(opt_time)
                
                # Generate response with optimized prompt
                optimized_response = shared_optimizer.generator.generate(optimized_prompt, max_length=100)
                
                # Evaluate both responses
                baseline_metrics = shared_optimizer.evaluator.evaluate(
                    prompt, baseline_response, task_type=task_type, reference=reference
                )
                
                optimized_metrics = shared_optimizer.evaluator.evaluate(
                    optimized_prompt, optimized_response, task_type=task_type, reference=reference
                )
                
                # Calculate improvements
                improvements = {}
                for metric, baseline_value in baseline_metrics.items():
                    if metric in optimized_metrics:
                        # Perplexity is better when lower
                        if metric == "perplexity":
                            if baseline_value > 0:
                                improvements[metric] = (baseline_value - optimized_metrics[metric]) / baseline_value
                        else:
                            improvements[metric] = optimized_metrics[metric] - baseline_value
                
                # Record result
                result = {
                    "original_prompt": prompt,
                    "optimized_prompt": optimized_prompt,
                    "original_response": baseline_response,
                    "optimized_response": optimized_response,
                    "task_type": task_type,
                    "baseline_metrics": baseline_metrics,
                    "optimized_metrics": optimized_metrics,
                    "improvements": improvements,
                    "optimization_time": opt_time
                }
                
                # Add reference if available
                if reference:
                    result["reference"] = reference
                
                # Store result
                results[strategy].append(result)
                
            except Exception as e:
                print(f"\nError evaluating prompt {i} with strategy {strategy}: {e}")
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Full evaluation results saved to {results_path}")
    except Exception as e:
        print(f"Error saving full results: {e}")
        # Try saving a simplified version
        simple_results = {}
        for strategy, strategy_results in results.items():
            simple_results[strategy] = []
            for result in strategy_results:
                # Create a simplified version with just strings and numbers
                simple_result = {
                    "original_prompt": result["original_prompt"],
                    "optimized_prompt": result["optimized_prompt"],
                    "task_type": result["task_type"],
                    "optimization_time": result["optimization_time"]
                }
                simple_results[strategy].append(simple_result)
        
        with open(results_path, 'w') as f:
            json.dump(simple_results, f, indent=2)

    # Calculate performance summary
    summary = calculate_performance_summary(results, optimization_times)
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "performance_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Performance summary saved to {summary_path}")
    
    # Generate visualization
    generate_evaluation_visualizations(results, summary, args.output_dir)
    
    return results, summary

def calculate_performance_summary(results, optimization_times):
    """Calculate performance summary from evaluation results."""
    summary = {
        "strategies": {},
        "overall": {},
        "task_performance": {},
        "metrics_comparison": {},
        "optimization_statistics": {}
    }
    
    # Calculate strategy-specific statistics
    for strategy, strategy_results in results.items():
        if not strategy_results:
            continue
        
        # Initialize strategy summary
        summary["strategies"][strategy] = {
            "num_samples": len(strategy_results),
            "avg_optimization_time": np.mean(optimization_times[strategy]),
            "improvement_rate": 0,
            "avg_improvement": 0,
            "metrics": {}
        }
        
        # Calculate improvement statistics
        improvement_count = 0
        total_improvement = 0
        metrics_improvements = {}
        
        for result in strategy_results:
            if "improvements" in result:
                # Check if there's any positive improvement
                any_improvement = False
                result_improvement = 0
                result_metric_count = 0
                
                for metric, value in result["improvements"].items():
                    # Initialize metric in dictionary if not present
                    if metric not in metrics_improvements:
                        metrics_improvements[metric] = []
                    
                    # Add improvement value
                    metrics_improvements[metric].append(value)
                    
                    # Calculate overall improvement excluding perplexity
                    if metric != "perplexity":
                        result_improvement += value
                        result_metric_count += 1
                        if value > 0:
                            any_improvement = True
                
                # Calculate average improvement for this result
                if result_metric_count > 0:
                    avg_result_improvement = result_improvement / result_metric_count
                    total_improvement += avg_result_improvement
                    
                    if any_improvement:
                        improvement_count += 1
        
        # Calculate improvement rate
        if strategy_results:
            summary["strategies"][strategy]["improvement_rate"] = improvement_count / len(strategy_results)
        
        # Calculate average overall improvement
        if improvement_count > 0:
            summary["strategies"][strategy]["avg_improvement"] = total_improvement / improvement_count
        
        # Calculate average metric-specific improvements
        for metric, values in metrics_improvements.items():
            if values:
                summary["strategies"][strategy]["metrics"][metric] = {
                    "avg_improvement": np.mean(values),
                    "max_improvement": np.max(values),
                    "min_improvement": np.min(values),
                    "std_improvement": np.std(values)
                }
        
        # Calculate task-specific performance
        task_performance = {}
        for result in strategy_results:
            task_type = result.get("task_type", "general")
            
            if task_type not in task_performance:
                task_performance[task_type] = {
                    "count": 0,
                    "improvements": []
                }
            
            # Add task-specific improvement
            if "improvements" in result:
                # Calculate overall improvement excluding perplexity
                task_improvement = 0
                task_metric_count = 0
                
                for metric, value in result["improvements"].items():
                    if metric != "perplexity":
                        task_improvement += value
                        task_metric_count += 1
                
                if task_metric_count > 0:
                    avg_task_improvement = task_improvement / task_metric_count
                    task_performance[task_type]["improvements"].append(avg_task_improvement)
                    task_performance[task_type]["count"] += 1
        
        # Calculate average task-specific improvement
        for task, data in task_performance.items():
            if data["improvements"]:
                avg_improvement = np.mean(data["improvements"])
                if task not in summary["task_performance"]:
                    summary["task_performance"][task] = {}
                summary["task_performance"][task][strategy] = {
                    "count": data["count"],
                    "avg_improvement": avg_improvement
                }
    
    # Calculate overall best strategy
    best_strategy = None
    best_improvement = -1
    
    for strategy, data in summary["strategies"].items():
        if data["avg_improvement"] > best_improvement:
            best_improvement = data["avg_improvement"]
            best_strategy = strategy
    
    summary["overall"]["best_strategy"] = best_strategy
    summary["overall"]["best_improvement"] = best_improvement
    
    # Calculate optimization time statistics
    for strategy, times in optimization_times.items():
        if times:
            summary["optimization_statistics"][strategy] = {
                "avg_time": np.mean(times),
                "max_time": np.max(times),
                "min_time": np.min(times),
                "std_time": np.std(times)
            }
    
    # Calculate metrics comparison
    all_metrics = set()
    for strategy, data in summary["strategies"].items():
        all_metrics.update(data.get("metrics", {}).keys())
    
    for metric in all_metrics:
        metric_comparison = {}
        for strategy, data in summary["strategies"].items():
            if "metrics" in data and metric in data["metrics"]:
                metric_comparison[strategy] = data["metrics"][metric]["avg_improvement"]
        
        if metric_comparison:
            summary["metrics_comparison"][metric] = metric_comparison
    
    return summary

def generate_evaluation_visualizations(results, summary, output_dir):
    """Generate visualizations for evaluation results."""
    print("\n=== Generating Evaluation Visualizations ===")
    
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Overall strategy comparison chart
    if "strategies" in summary:
        strategies = list(summary["strategies"].keys())
        improvements = [summary["strategies"][s]["avg_improvement"] for s in strategies]
        
        if strategies and improvements:
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.bar(strategies, improvements, color=promptsage_colors[:len(strategies)])
            
            # Add value labels
            for bar, value in zip(bars, improvements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_xlabel("Optimization Strategy")
            ax.set_ylabel("Average Improvement")
            ax.set_title("Overall Strategy Performance Comparison", fontweight='bold', fontsize=18)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            plt.tight_layout()
            fig_path = os.path.join(viz_dir, "strategy_performance.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Overall strategy comparison saved to {fig_path}")
    
    # 2. Metrics comparison charts
    if "metrics_comparison" in summary:
        for metric, comparison in summary["metrics_comparison"].items():
            if comparison:
                strategies = list(comparison.keys())
                values = list(comparison.values())
                
                fig, ax = plt.subplots(figsize=(12, 7))
                bars = ax.bar(strategies, values, color=promptsage_colors[:len(strategies)])
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                # Format metric name for title
                metric_title = metric.replace("_", " ").title()
                
                ax.set_xlabel("Optimization Strategy")
                ax.set_ylabel(f"{metric_title} Improvement")
                ax.set_title(f"{metric_title} Improvement by Strategy", fontweight='bold', fontsize=18)
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                plt.tight_layout()
                fig_path = os.path.join(viz_dir, f"metric_{metric}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Metric comparison for {metric} saved to {fig_path}")
    
    # 3. Task-specific performance heatmap
    if "task_performance" in summary:
        tasks = list(summary["task_performance"].keys())
        strategies = set()
        
        for task_data in summary["task_performance"].values():
            strategies.update(task_data.keys())
        
        strategies = list(strategies)
        
        if tasks and strategies:
            # Create data matrix
            data_matrix = np.zeros((len(tasks), len(strategies)))
            
            for i, task in enumerate(tasks):
                for j, strategy in enumerate(strategies):
                    if strategy in summary["task_performance"][task]:
                        data_matrix[i, j] = summary["task_performance"][task][strategy]["avg_improvement"]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Create heatmap
            sns.heatmap(
                data_matrix, 
                annot=True, 
                fmt=".4f", 
                cmap=promptsage_cmap, 
                xticklabels=strategies,
                yticklabels=tasks,
                linewidths=0.5,
                ax=ax
            )
            
            ax.set_title("Task-Specific Optimization Performance", fontweight='bold', fontsize=18)
            ax.set_xlabel("Strategy", fontsize=14)
            ax.set_ylabel("Task Type", fontsize=14)
            
            plt.tight_layout()
            fig_path = os.path.join(viz_dir, "task_performance_heatmap.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Task performance heatmap saved to {fig_path}")
    
    # 4. Optimization time comparison
    if "optimization_statistics" in summary:
        strategies = list(summary["optimization_statistics"].keys())
        times = [summary["optimization_statistics"][s]["avg_time"] for s in strategies]
        std_times = [summary["optimization_statistics"][s]["std_time"] for s in strategies]
        
        if strategies and times:
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.bar(
                strategies, 
                times, 
                yerr=std_times, 
                capsize=10, 
                color=promptsage_colors[:len(strategies)],
                alpha=0.8
            )
            
            # Add value labels
            for bar, value in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{value:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_xlabel("Optimization Strategy")
            ax.set_ylabel("Average Optimization Time (seconds)")
            ax.set_title("Strategy Optimization Time Comparison", fontweight='bold', fontsize=18)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            plt.tight_layout()
            fig_path = os.path.join(viz_dir, "optimization_time.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Optimization time comparison saved to {fig_path}")
    
    # 5. Improvement rate comparison
    if "strategies" in summary:
        strategies = list(summary["strategies"].keys())
        rates = [summary["strategies"][s]["improvement_rate"] for s in strategies]
        
        if strategies and rates:
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.bar(strategies, rates, color=promptsage_colors[:len(strategies)])
            
            # Add value labels
            for bar, value in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{value:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_xlabel("Optimization Strategy")
            ax.set_ylabel("Improvement Rate")
            ax.set_title("Strategy Improvement Rate Comparison", fontweight='bold', fontsize=18)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            ax.set_ylim(0, max(rates) * 1.2)  # Add some headroom for labels
            
            plt.tight_layout()
            fig_path = os.path.join(viz_dir, "improvement_rate.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Improvement rate comparison saved to {fig_path}")
    
    # 6. Create radar chart for multi-metric comparison
    if "metrics_comparison" in summary:
        # Collect all metrics and strategies
        metrics = list(summary["metrics_comparison"].keys())
        strategies = set()
        for metric_data in summary["metrics_comparison"].values():
            strategies.update(metric_data.keys())
        strategies = list(strategies)
        
        if len(metrics) >= 3 and strategies:
            # Prepare data for radar chart
            strategy_values = {}
            
            for strategy in strategies:
                values = []
                for metric in metrics:
                    if strategy in summary["metrics_comparison"][metric]:
                        values.append(summary["metrics_comparison"][metric][strategy])
                    else:
                        values.append(0)
                
                strategy_values[strategy] = values
            
            # Create radar chart
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of metrics
            N = len(metrics)
            
            # Angle for each metric (evenly spaced)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Format metric labels
            metric_labels = [m.replace("_", " ").title() for m in metrics]
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=12)
            
            # Plot each strategy
            for i, (strategy, values) in enumerate(strategy_values.items()):
                values += values[:1]  # Close the loop
                ax.plot(angles, values, linewidth=3, label=strategy, color=promptsage_colors[i % len(promptsage_colors)])
                ax.fill(angles, values, alpha=0.25, color=promptsage_colors[i % len(promptsage_colors)])
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
            
            # Set title
            plt.title("Multi-Metric Strategy Comparison", fontweight='bold', fontsize=20, pad=20)
            
            plt.tight_layout()
            fig_path = os.path.join(viz_dir, "strategy_radar_chart.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Multi-metric radar chart saved to {fig_path}")
    
    # 7. Generate prompt improvement examples
    generate_improvement_examples(results, viz_dir)
    
    print(f"All evaluation visualizations saved to {viz_dir}")

def generate_improvement_examples(results, viz_dir):
    """Generate visualizations for prompt improvement examples."""
    print("\n=== Generating Prompt Improvement Examples ===")
    
    # Create examples directory
    examples_dir = os.path.join(viz_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    
    # Collect successful optimizations
    successful_examples = []
    
    for strategy, strategy_results in results.items():
        for result in strategy_results:
            # Check if there's improvement
            if "improvements" in result:
                # Calculate overall improvement excluding perplexity
                total_improvement = 0
                improvement_count = 0
                
                for metric, value in result["improvements"].items():
                    if metric != "perplexity" and value > 0:
                        total_improvement += value
                        improvement_count += 1
                
                # Only include examples with positive improvement
                if improvement_count > 0 and total_improvement > 0:
                    avg_improvement = total_improvement / improvement_count
                    
                    # Add to successes
                    successful_examples.append({
                        "strategy": strategy,
                        "original_prompt": result["original_prompt"],
                        "optimized_prompt": result["optimized_prompt"],
                        "original_response": result["original_response"],
                        "optimized_response": result["optimized_response"],
                        "task_type": result.get("task_type", "general"),
                        "improvement_score": avg_improvement,
                        "improvements": result["improvements"]
                    })
    
    # Sort by improvement score
    successful_examples.sort(key=lambda x: x["improvement_score"], reverse=True)
    
    # Take top examples
    top_examples = successful_examples[:min(10, len(successful_examples))]
    
    if top_examples:
        # Create examples markdown file
        examples_md_path = os.path.join(examples_dir, "optimization_examples.md")
        with open(examples_md_path, 'w') as f:
            f.write("# PromptSage Optimization Examples\n\n")
            
            for i, example in enumerate(top_examples):
                f.write(f"## Example {i+1}: {example['task_type']} Task\n\n")
                f.write(f"**Strategy:** {example['strategy']}\n\n")
                f.write(f"**Improvement Score:** {example['improvement_score']:.4f}\n\n")
                f.write("**Original Prompt:**\n```\n")
                f.write(f"{example['original_prompt']}\n")
                f.write("```\n\n")
                f.write("**Optimized Prompt:**\n```\n")
                f.write(f"{example['optimized_prompt']}\n")
                f.write("```\n\n")
                f.write("**Original Response:**\n```\n")
                f.write(f"{example['original_response']}\n")
                f.write("```\n\n")
                f.write("**Optimized Response:**\n```\n")
                f.write(f"{example['optimized_response']}\n")
                f.write("```\n\n")
                
                # Add metrics table
                f.write("**Metrics Improvement:**\n\n")
                f.write("| Metric | Improvement |\n")
                f.write("|--------|-------------|\n")
                
                for metric, value in example["improvements"].items():
                    f.write(f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n")
                
                f.write("\n---\n\n")
        
        print(f"Created {len(top_examples)} optimization examples in {examples_md_path}")
        
        # Create visualization of example improvements
        example_ids = [f"Ex {i+1}" for i in range(len(top_examples))]
        improvements = [ex["improvement_score"] for ex in top_examples]
        strategies = [ex["strategy"] for ex in top_examples]
        
        # Create colormap based on strategies
        strategy_set = list(set(strategies))
        strategy_colors = {strategy: promptsage_colors[i % len(promptsage_colors)] 
                         for i, strategy in enumerate(strategy_set)}
        bar_colors = [strategy_colors[s] for s in strategies]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(example_ids, improvements, color=bar_colors)
        
        # Add strategy annotations to bars
        for i, (bar, strategy) in enumerate(zip(bars, strategies)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                strategy,
                ha='center', va='bottom', 
                fontsize=10, rotation=45,
                fontweight='bold'
            )
            
            # Add improvement value
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height/2,
                f"{improvements[i]:.4f}",
                ha='center', va='center', 
                fontsize=11, color='white',
                fontweight='bold'
            )
        
        # Set title and labels
        ax.set_title("Improvement Scores for Top Examples", fontweight='bold', fontsize=18)
        ax.set_xlabel("Examples", fontsize=14)
        ax.set_ylabel("Improvement Score", fontsize=14)
        
        # Add grid for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend for strategies
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=strategy) 
                          for strategy, color in strategy_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        fig_path = os.path.join(examples_dir, "top_examples_improvement.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Top examples visualization saved to {fig_path}")
        
        # Create task distribution pie chart for examples
        task_counts = {}
        for example in top_examples:
            task = example["task_type"]
            task_counts[task] = task_counts.get(task, 0) + 1
        
        if task_counts:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                task_counts.values(), 
                labels=task_counts.keys(),
                autopct='%1.1f%%',
                textprops={'fontsize': 14, 'fontweight': 'bold'},
                colors=promptsage_colors[:len(task_counts)]
            )
            
            # Customize text
            for autotext in autotexts:
                autotext.set_fontweight('bold')
            
            ax.set_title("Task Distribution in Top Examples", fontweight='bold', fontsize=18)
            
            plt.tight_layout()
            fig_path = os.path.join(examples_dir, "task_distribution.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Task distribution visualization saved to {fig_path}")
    
    else:
        print("No successful optimization examples found")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment
    device = setup_environment(args)
    
    # Prepare datasets
    training_data, evaluation_data = prepare_datasets(args, device)
    
    # Load model
    model, tokenizer = load_model(args.model, device, args.quick_mode)
    
    # Train meta-learning optimizer
    meta_optimizer, meta_model_path, training_metrics = train_meta_optimizer(
        training_data, model, tokenizer, args, device
    )
    
    # Evaluate optimization strategies
    results, summary = evaluate_optimization_strategies(
        evaluation_data, model, tokenizer, meta_model_path, args, device
    )
    
    print("\n=== Research Evaluation Complete ===")
    print(f"All results saved to {args.output_dir}")
    print(f"Visualizations saved to {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()