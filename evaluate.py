"""
Evaluation module for PromptSage
"""

import os
import argparse
import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import time

from promptsage import PromptOptimizer
from promptsage.utils.visualization import plot_comparison, plot_model_comparison

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PromptSage optimization")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="EleutherAI/gpt-neo-1.3B", 
        help="Model name or path"
    )
    parser.add_argument(
        "--test_data", 
        type=str, 
        required=True, 
        help="Path to test prompts (CSV or JSON)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./evaluation_results", 
        help="Output directory for results"
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="auto", 
        choices=["auto", "evolution", "meta", "contrastive"],
        help="Optimization strategy to evaluate"
    )
    parser.add_argument(
        "--num_iterations", 
        type=int, 
        default=5, 
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--compare_original", 
        action="store_true", 
        help="Compare with original prompt"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Device to use (cpu, cuda, auto)"
    )
    
    return parser.parse_args()

def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """Load test prompts from file."""
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        # Convert DataFrame to list of dicts
        test_prompts = df.to_dict(orient='records')
    elif data_path.endswith(".json"):
        with open(data_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            test_prompts = data
        else:
            # If it's a dict structure, try to extract prompts
            if "prompts" in data and isinstance(data["prompts"], list):
                test_prompts = data["prompts"]
            else:
                raise ValueError("JSON format not supported. Expected list or dict with 'prompts' key.")
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")
        
    return test_prompts

def evaluate_optimizer(
    test_prompts: List[Dict[str, Any]],
    model_name: str,
    strategy: str,
    num_iterations: int,
    device: str = "auto",
    meta_learner_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Evaluate prompt optimization on test prompts."""
    # Initialize optimizer
    optimizer = PromptOptimizer(
        model_name=model_name,
        optimization_strategy=strategy,
        device=device
    )
    
    # Load meta-learner if specified
    if meta_learner_path and os.path.exists(meta_learner_path) and strategy == "meta":
        meta_learner = optimizer.optimizers.get("meta")
        if meta_learner:
            print(f"Loading meta-learner from {meta_learner_path}")
            meta_learner = meta_learner.__class__(model_path=meta_learner_path)
            optimizer.optimizers["meta"] = meta_learner
    
    results = []
    
    # Evaluate each prompt
    for i, prompt_data in enumerate(test_prompts):
        print(f"Optimizing prompt {i+1}/{len(test_prompts)}")
        
        # Extract prompt and additional data
        if "prompt" in prompt_data:
            original_prompt = prompt_data["prompt"]
        elif "original_prompt" in prompt_data:
            original_prompt = prompt_data["original_prompt"]
        else:
            print(f"Skipping prompt {i+1}: no prompt found")
            continue
        
        # Extract or default task type
        task_type = prompt_data.get("task_type", "general")
        
        # Extract reference if available
        reference = prompt_data.get("reference", None)
        
        # Extract or default optimization goals
        optimization_goals = prompt_data.get("optimization_goals", ["clarity", "specificity", "relevance"])
        
        # Time the optimization
        start_time = time.time()
        
        # Optimize the prompt
        try:
            optimized_prompt, metrics = optimizer.optimize(
                prompt=original_prompt,
                task_type=task_type,
                optimization_goals=optimization_goals,
                num_iterations=num_iterations
            )
            
            optimization_time = time.time() - start_time
            
            # If reference available, evaluate against it
            if reference:
                optimized_response = optimizer.generator.generate(optimized_prompt)
                reference_metrics = optimizer.evaluator.evaluate(
                    optimized_prompt, optimized_response, reference=reference
                )
                metrics["reference_metrics"] = reference_metrics
            
            # Create result entry
            result = {
                "original_prompt": original_prompt,
                "optimized_prompt": optimized_prompt,
                "task_type": task_type,
                "optimization_goals": optimization_goals,
                "metrics": metrics,
                "optimization_time": optimization_time
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error optimizing prompt {i+1}: {e}")
            continue
    
    return results

def generate_reports(
    results: List[Dict[str, Any]],
    output_dir: str,
    compare_original: bool = False
):
    """Generate evaluation reports from results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save raw results JSON
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # 2. Create summary metrics
    summary = {
        "total_prompts": len(results),
        "avg_optimization_time": sum(r.get("optimization_time", 0) for r in results) / max(1, len(results)),
        "metrics_improvement": {}
    }
    
    # Calculate average improvement for each metric
    all_metrics = set()
    for result in results:
        metrics = result.get("metrics", {})
        if "baseline" in metrics and "optimized" in metrics:
            baseline = metrics["baseline"]
            optimized = metrics["optimized"]
            
            # Collect all possible metrics
            all_metrics.update(baseline.keys())
            all_metrics.update(optimized.keys())
    
    # Calculate average improvement
    for metric in all_metrics:
        improvements = []
        for result in results:
            metrics = result.get("metrics", {})
            if "baseline" in metrics and "optimized" in metrics:
                baseline = metrics["baseline"]
                optimized = metrics["optimized"]
                
                if metric in baseline and metric in optimized:
                    # For perplexity, lower is better
                    if metric == "perplexity":
                        if baseline[metric] > 0 and optimized[metric] > 0:
                            # Calculate relative improvement (negative is better)
                            improvement = (optimized[metric] - baseline[metric]) / baseline[metric]
                            improvements.append(improvement)
                    else:
                        # For other metrics, higher is better
                        improvement = optimized[metric] - baseline[metric]
                        improvements.append(improvement)
        
        if improvements:
            summary["metrics_improvement"][metric] = sum(improvements) / len(improvements)
    
    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # 3. Generate visualizations
    if compare_original:
        # For each result, compare original and optimized
        for i, result in enumerate(results):
            # Generate comparison visualizations
            comparison_data = {
                "original_prompt": result["original_prompt"],
                "optimized_prompt": result["optimized_prompt"]
            }
            
            # Add metrics if available
            if "metrics" in result and "baseline" in result["metrics"] and "optimized" in result["metrics"]:
                comparison_data["original_metrics"] = result["metrics"]["baseline"]
                comparison_data["optimized_metrics"] = result["metrics"]["optimized"]
            
            # Generate original and optimized responses
            try:
                optimizer = PromptOptimizer(model_name=None)  # Lightweight instance
                original_response = result.get("metrics", {}).get("original_response", "No response available")
                optimized_response = result.get("metrics", {}).get("optimized_response", "No response available")
                
                comparison_data["original_response"] = original_response
                comparison_data["optimized_response"] = optimized_response
                
                # Generate visualizations
                visualizations = plot_comparison(comparison_data)
                
                # Save each visualization
                for viz_name, viz_data in visualizations.items():
                    if viz_data["type"] == "image":
                        img_data = viz_data["data"]
                        import base64
                        img_bytes = base64.b64decode(img_data)
                        
                        # Save image
                        img_path = os.path.join(output_dir, f"result_{i}_{viz_name}.png")
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                
            except Exception as e:
                print(f"Error generating visualizations for result {i}: {e}")
    
    # 4. Generate aggregate visualizations
    try:
        # Prepare data for model comparison visualization
        model_data = {
            "Original": {},
            "Optimized": {}
        }
        
        # Extract metrics
        for result in results:
            if "metrics" in result and "baseline" in result["metrics"] and "optimized" in result["metrics"]:
                baseline = result["metrics"]["baseline"]
                optimized = result["metrics"]["optimized"]
                
                # Aggregate each metric
                for metric, value in baseline.items():
                    if metric not in model_data["Original"]:
                        model_data["Original"][metric] = []
                    model_data["Original"][metric].append(value)
                
                for metric, value in optimized.items():
                    if metric not in model_data["Optimized"]:
                        model_data["Optimized"][metric] = []
                    model_data["Optimized"][metric].append(value)
        
        # Calculate averages
        for model in model_data:
            for metric in model_data[model]:
                values = model_data[model][metric]
                if values:
                    model_data[model][metric] = sum(values) / len(values)
        
        # Generate visualizations
        visualizations = plot_model_comparison(model_data)
        
        # Save each visualization
        for viz_name, viz_data in visualizations.items():
            if viz_data["type"] == "image":
                img_data = viz_data["data"]
                import base64
                img_bytes = base64.b64decode(img_data)
                
                # Save image
                img_path = os.path.join(output_dir, f"aggregate_{viz_name}.png")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                    
    except Exception as e:
        print(f"Error generating aggregate visualizations: {e}")

def main():
    args = parse_args()
    
    # Load test data
    test_prompts = load_test_data(args.test_data)
    print(f"Loaded {len(test_prompts)} test prompts from {args.test_data}")
    
    # Evaluate optimization
    results = evaluate_optimizer(
        test_prompts=test_prompts,
        model_name=args.model_name,
        strategy=args.strategy,
        num_iterations=args.num_iterations,
        device=args.device
    )
    
    # Generate reports
    generate_reports(
        results=results,
        output_dir=args.output_dir,
        compare_original=args.compare_original
    )
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()