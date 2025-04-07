"""
Visualization utilities for PromptSage
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import io
import base64

def plot_comparison(comparison_data: Dict) -> Dict:
    """
    Create visualizations comparing original and optimized prompts.
    
    Args:
        comparison_data: Dictionary with comparison metrics
        
    Returns:
        Dictionary of visualization objects/data
    """
    visualizations = {}
    
    # Extract relevant data
    original_metrics = comparison_data.get("original_metrics", {})
    optimized_metrics = comparison_data.get("optimized_metrics", {})
    
    # Common metrics to plot
    common_metrics = list(set(original_metrics.keys()).intersection(optimized_metrics.keys()))
    common_metrics = [m for m in common_metrics if isinstance(original_metrics[m], (int, float))]
    
    if common_metrics:
        # 1. Bar chart comparing metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df = pd.DataFrame({
            'Original': [original_metrics.get(m, 0) for m in common_metrics],
            'Optimized': [optimized_metrics.get(m, 0) for m in common_metrics]
        }, index=common_metrics)
        
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title('Comparison of Metrics: Original vs Optimized Prompt')
        ax.set_ylabel('Score')
        ax.set_xlabel('Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Add to visualizations
        visualizations['metrics_comparison'] = {
            'type': 'image',
            'format': 'png',
            'data': base64.b64encode(buf.read()).decode('utf-8')
        }
        plt.close(fig)
    
    # 2. Word clouds for prompt comparison
    try:
        from wordcloud import WordCloud
        
        original_prompt = comparison_data.get("original_prompt", "")
        optimized_prompt = comparison_data.get("optimized_prompt", "")
        
        # Create word clouds
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original prompt word cloud
        wordcloud_orig = WordCloud(width=800, height=400, background_color='white').generate(original_prompt)
        ax1.imshow(wordcloud_orig, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title('Original Prompt')
        
        # Optimized prompt word cloud
        wordcloud_opt = WordCloud(width=800, height=400, background_color='white').generate(optimized_prompt)
        ax2.imshow(wordcloud_opt, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title('Optimized Prompt')
        
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Add to visualizations
        visualizations['wordcloud_comparison'] = {
            'type': 'image',
            'format': 'png',
            'data': base64.b64encode(buf.read()).decode('utf-8')
        }
        plt.close(fig)
        
    except ImportError:
        # WordCloud not available
        pass
    
    # 3. Response length comparison
    original_response = comparison_data.get("original_response", "")
    optimized_response = comparison_data.get("optimized_response", "")
    
    orig_len = len(original_response.split())
    opt_len = len(optimized_response.split())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=['Original', 'Optimized'], y=[orig_len, opt_len], ax=ax)
    ax.set_title('Response Length Comparison')
    ax.set_ylabel('Word Count')
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Add to visualizations
    visualizations['length_comparison'] = {
        'type': 'image',
        'format': 'png',
        'data': base64.b64encode(buf.read()).decode('utf-8')
    }
    plt.close(fig)
    
    return visualizations

def plot_optimization_history(history: List[Dict]) -> Dict:
    """
    Create visualizations of optimization history.
    
    Args:
        history: List of dictionaries containing optimization history
        
    Returns:
        Dictionary of visualization objects/data
    """
    visualizations = {}
    
    # Extract data
    iterations = [h.get("iteration", i) for i, h in enumerate(history)]
    scores = [h.get("best_score", 0) for h in history]
    avg_scores = [h.get("avg_score", 0) for h in history if "avg_score" in h]
    
    # 1. Plot score progression
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, scores, marker='o', label='Best Score')
    
    if avg_scores and len(avg_scores) == len(iterations):
        ax.plot(iterations, avg_scores, marker='x', linestyle='--', label='Average Score')
    
    ax.set_title('Optimization Score Progression')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Add to visualizations
    visualizations['score_progression'] = {
        'type': 'image',
        'format': 'png',
        'data': base64.b64encode(buf.read()).decode('utf-8')
    }
    plt.close(fig)
    
    # 2. Plot score distribution if we have candidate scores
    candidate_scores = []
    for h in history:
        if "candidates" in h and h["candidates"]:
            for _, _, score in h["candidates"]:
                if isinstance(score, (int, float)):
                    candidate_scores.append(score)
    
    if candidate_scores:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(candidate_scores, kde=True, ax=ax)
        ax.set_title('Distribution of Candidate Scores')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Add to visualizations
        visualizations['score_distribution'] = {
            'type': 'image',
            'format': 'png',
            'data': base64.b64encode(buf.read()).decode('utf-8')
        }
        plt.close(fig)
        
    # 3. Plot prompt evolution metrics if available
    evolution_metrics = {}
    for h in history:
        prompt = h.get("best_prompt", "")
        if prompt:
            evolution_metrics[h.get("iteration", 0)] = {
                "length": len(prompt.split()),
                "question_mark": 1 if "?" in prompt else 0,
                "specificity": prompt.lower().count("specific") + prompt.lower().count("detail"),
                "examples": prompt.lower().count("example")
            }
    
    if evolution_metrics:
        df = pd.DataFrame.from_dict(evolution_metrics, orient='index')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics = ["length", "question_mark", "specificity", "examples"]
        titles = ["Prompt Length", "Question Format", "Specificity Words", "Example Requests"]
        
        for i, metric in enumerate(metrics):
            df[metric].plot(kind='line', marker='o', ax=axes[i])
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Iteration')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Add to visualizations
        visualizations['prompt_evolution'] = {
            'type': 'image',
            'format': 'png',
            'data': base64.b64encode(buf.read()).decode('utf-8')
        }
        plt.close(fig)
    
    return visualizations

def plot_model_comparison(
    model_results: Dict[str, Dict[str, Union[float, List[float]]]]
) -> Dict:
    """
    Create visualizations comparing performance across models.
    
    Args:
        model_results: Dictionary with model names as keys and metrics as values
        
    Returns:
        Dictionary of visualization objects/data
    """
    visualizations = {}
    
    # Extract data
    models = list(model_results.keys())
    
    if not models:
        return visualizations
        
    # Find common metrics across all models
    all_metrics = set()
    for model, metrics in model_results.items():
        all_metrics.update(metrics.keys())
    
    # Filter to numeric metrics only
    numeric_metrics = []
    for metric in all_metrics:
        is_numeric = all(
            isinstance(model_results[model].get(metric), (int, float, list)) 
            for model in models if metric in model_results[model]
        )
        if is_numeric:
            numeric_metrics.append(metric)
    
    # 1. Bar chart for each metric
    for metric in numeric_metrics:
        # Extract values for this metric
        values = []
        for model in models:
            if metric in model_results[model]:
                val = model_results[model][metric]
                # If it's a list, use the mean
                if isinstance(val, list):
                    val = sum(val) / len(val)
                values.append(val)
            else:
                values.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=models, y=values, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison Across Models')
        ax.set_ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Add to visualizations
        visualizations[f'{metric}_comparison'] = {
            'type': 'image',
            'format': 'png',
            'data': base64.b64encode(buf.read()).decode('utf-8')
        }
        plt.close(fig)
    
    # 2. Radar chart for multi-metric comparison if we have enough metrics
    if len(numeric_metrics) >= 3:
        # Prepare data
        model_data = {}
        for model in models:
            model_data[model] = []
            for metric in numeric_metrics:
                if metric in model_results[model]:
                    val = model_results[model][metric]
                    # If it's a list, use the mean
                    if isinstance(val, list):
                        val = sum(val) / len(val)
                    model_data[model].append(val)
                else:
                    model_data[model].append(0)
        
        # Normalize data
        for metric_idx in range(len(numeric_metrics)):
            values = [model_data[model][metric_idx] for model in models]
            max_val = max(values) if values else 1
            for model in models:
                if max_val > 0:
                    model_data[model][metric_idx] /= max_val
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Compute angle for each metric
        angles = np.linspace(0, 2*np.pi, len(numeric_metrics), endpoint=False).tolist()
        # Close the loop
        angles += angles[:1]
        
        # Plot each model
        for model in models:
            values = model_data[model]
            # Close the loop
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(numeric_metrics)
        ax.set_title("Model Performance Comparison")
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Add to visualizations
        visualizations['radar_comparison'] = {
            'type': 'image',
            'format': 'png',
            'data': base64.b64encode(buf.read()).decode('utf-8')
        }
        plt.close(fig)
    
    return visualizations