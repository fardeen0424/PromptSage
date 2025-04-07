"""
Training module for PromptSage
"""

import os
import argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import json
from datasets import Dataset
from typing import List, Dict, Any, Optional

from promptsage import PromptOptimizer
from promptsage.models.meta_learner import MetaLearningOptimizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train PromptSage models")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="EleutherAI/gpt-neo-1.3B", 
        help="Model name or path"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to training data (CSV or JSON)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./saved_models", 
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--optimizer_type", 
        type=str, 
        default="meta", 
        choices=["meta", "evolution", "contrastive"],
        help="Optimizer type to train"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5, 
        help="Learning rate"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Device to use (cpu, cuda, auto)"
    )
    
    return parser.parse_args()

def load_training_data(data_path: str) -> pd.DataFrame:
    """Load training data from file."""
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    elif data_path.endswith(".json"):
        with open(data_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            # If it's a dict structure, try to convert
            if "prompts" in data and isinstance(data["prompts"], list):
                return pd.DataFrame(data["prompts"])
            else:
                raise ValueError("JSON format not supported. Expected list or dict with 'prompts' key.")
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")

def create_dataset_from_pairs(prompt_pairs: List[Dict[str, str]]) -> Dataset:
    """Create a Hugging Face dataset from prompt pairs."""
    # Extract original and optimized prompts
    original_prompts = [pair["original_prompt"] for pair in prompt_pairs]
    optimized_prompts = [pair["optimized_prompt"] for pair in prompt_pairs]
    
    # Create dataset
    dataset_dict = {
        "original_prompt": original_prompts,
        "optimized_prompt": optimized_prompts
    }
    
    # Add task types and metrics if available
    if all("task_type" in pair for pair in prompt_pairs):
        dataset_dict["task_type"] = [pair["task_type"] for pair in prompt_pairs]
    
    if all("metrics" in pair for pair in prompt_pairs):
        dataset_dict["metrics"] = [json.dumps(pair["metrics"]) for pair in prompt_pairs]
    
    return Dataset.from_dict(dataset_dict)

def train_meta_learner(
    data: pd.DataFrame,
    output_dir: str,
    model_name: str = "EleutherAI/gpt-neo-1.3B",
    device: str = "auto"
):
    """Train a meta-learning optimizer using example prompt pairs."""
    # Initialize optimizer with model
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Loading model {model_name} on {device}...")
    # We need model and tokenizer for evaluation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Create prompt optimizer
    prompt_optimizer = PromptOptimizer(
        model_name=model_name,
        optimization_strategy="meta",
        device=device
    )
    
    # Extract meta learner 
    meta_learner = prompt_optimizer.optimizers["meta"]
    
    # Create prompt pairs from data
    prompt_pairs = []
    
    for idx, row in data.iterrows():
        # Check required columns
        if "original_prompt" not in row or "optimized_prompt" not in row:
            print(f"Skipping row {idx}: missing required columns")
            continue
            
        # Create a prompt pair entry
        pair = {
            "original_prompt": row["original_prompt"],
            "optimized_prompt": row["optimized_prompt"]
        }
        
        # Add task type if available
        if "task_type" in row:
            pair["task_type"] = row["task_type"]
        else:
            # Infer task type from prompt
            analysis = prompt_optimizer.analyzer.analyze(row["original_prompt"])
            inferred_tasks = analysis["likely_tasks"]
            pair["task_type"] = inferred_tasks[0] if inferred_tasks else "general"
        
        # Generate evaluation metrics if not provided
        if "metrics" not in row:
            original_response = prompt_optimizer.generator.generate(row["original_prompt"])
            optimized_response = prompt_optimizer.generator.generate(row["optimized_prompt"])
            
            original_metrics = prompt_optimizer.evaluator.evaluate(
                row["original_prompt"], original_response, task_type=pair["task_type"]
            )
            optimized_metrics = prompt_optimizer.evaluator.evaluate(
                row["optimized_prompt"], optimized_response, task_type=pair["task_type"]
            )
            
            # Calculate improvement
            improvement = {}
            for key in original_metrics:
                if key in optimized_metrics:
                    improvement[key] = optimized_metrics[key] - original_metrics[key]
            
            pair["metrics"] = {
                "original": original_metrics,
                "optimized": optimized_metrics,
                "improvement": improvement
            }
        
        prompt_pairs.append(pair)
    
    print(f"Created {len(prompt_pairs)} prompt pairs for training")
    
    # Learn from these pairs
    for pair in prompt_pairs:
        # Calculate a score diff based on improvements
        score_diff = 0.0
        
        if "metrics" in pair and "improvement" in pair["metrics"]:
            improvements = pair["metrics"]["improvement"]
            # Average all improvements
            if improvements:
                score_diff = sum(improvements.values()) / len(improvements)
                
        # If no explicit metrics, use a default score diff
        if score_diff == 0.0:
            score_diff = 0.2  # Assume some improvement
        
        # Add to contrastive pairs for learning
        meta_learner.contrastive_pairs.append((
            pair["optimized_prompt"],
            pair["original_prompt"],
            score_diff
        ))
        
        # Also add transformation memory
        task_type = pair.get("task_type", "general")
        for transform_name in meta_learner.transformation_templates.keys():
            meta_learner._update_transformation_memory(
                task_type, transform_name, score_diff
            )
    
    # Update weights
    meta_learner._learn_from_contrastive_pairs()
    
    # Save the trained meta-learner
    os.makedirs(output_dir, exist_ok=True)
    meta_learner_path = os.path.join(output_dir, "meta_learner.json")
    meta_learner.save_model(meta_learner_path)
    
    print(f"Meta-learner trained and saved to {meta_learner_path}")
    
    return meta_learner_path

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training data
    data = load_training_data(args.data_path)
    print(f"Loaded {len(data)} prompt examples from {args.data_path}")
    
    # Choose training method based on optimizer type
    if args.optimizer_type == "meta":
        output_path = train_meta_learner(
            data=data,
            output_dir=args.output_dir,
            model_name=args.model_name,
            device=args.device
        )
        print(f"Training complete. Model saved at: {output_path}")
    else:
        print(f"Training for {args.optimizer_type} optimizer not implemented yet.")
        # Placeholder for other optimizer training methods

if __name__ == "__main__":
    main()