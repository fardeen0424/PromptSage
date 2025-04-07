"""
Core optimizer module for PromptSage - handles the main optimization pipeline
"""

import logging
from typing import Dict, List, Tuple, Union, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.metrics import calculate_perplexity, calculate_response_metrics
from .analyzer import PromptAnalyzer
from .evaluator import PromptEvaluator
from .generator import PromptGenerator
from ..models.evolution import EvolutionaryOptimizer
from ..models.meta_learner import MetaLearningOptimizer
from ..models.contrastive import ContrastiveOptimizer

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """Main class for optimizing prompts using various strategies"""
    
    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-1.3B",
        optimization_strategy: str = "auto",
        device: str = None,
        max_length: int = 100,
        temperature: float = 0.7,
        seed: int = 42,
    ):
        """
        Initialize the PromptOptimizer with specified parameters.
        
        Args:
            model_name: The name of the model to use for evaluation
            optimization_strategy: Strategy to use ('evolution', 'meta', 'contrastive', 'auto')
            device: Device to run the model on ('cpu', 'cuda', 'auto')
            max_length: Maximum length for generated responses
            temperature: Sampling temperature for generation
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.strategy = optimization_strategy
        self.max_length = max_length
        self.temperature = temperature
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Initialize tokenizer and model
        logger.info(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Initialize components
        self.analyzer = PromptAnalyzer()
        self.evaluator = PromptEvaluator(self.model, self.tokenizer, device=self.device)
        self.generator = PromptGenerator(self.model, self.tokenizer, device=self.device)
        
        # Initialize strategy-specific optimizers
        self.optimizers = {
            'evolution': EvolutionaryOptimizer(),
            'meta': MetaLearningOptimizer(),
            'contrastive': ContrastiveOptimizer(),
        }
        
    def optimize(
        self,
        prompt: str,
        task_type: str = "general",
        target_audience: str = "general",
        optimization_goals: List[str] = None,
        num_iterations: int = 10,
        verbose: bool = False,
    ) -> Tuple[str, Dict]:
        """
        Optimize the given prompt based on specified parameters.
        
        Args:
            prompt: The original prompt to optimize
            task_type: The type of task (explanation, code, story, etc.)
            target_audience: The target audience (beginners, experts, etc.)
            optimization_goals: Goals to prioritize (clarity, specificity, etc.)
            num_iterations: Number of iterations for optimization
            verbose: Whether to print progress information
            
        Returns:
            Tuple containing (optimized_prompt, metrics_dict)
        """
        if optimization_goals is None:
            optimization_goals = ["clarity", "specificity", "relevance"]
            
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
            
        logger.info(f"Starting prompt optimization for task type: {task_type}")
        logger.info(f"Original prompt: {prompt}")
        
        # Analyze the original prompt
        analysis = self.analyzer.analyze(prompt)
        logger.info(f"Initial analysis: {analysis}")
        
        # Select optimization strategy
        if self.strategy == "auto":
            # Choose best strategy based on task type and prompt analysis
            if task_type in ["explanation", "definition"]:
                strategy = "meta"
            elif task_type in ["creative", "story"]:
                strategy = "evolution"
            else:
                strategy = "contrastive"
            logger.info(f"Auto-selected strategy: {strategy}")
        else:
            strategy = self.strategy
            
        # Get appropriate optimizer
        optimizer = self.optimizers.get(strategy)
        if optimizer is None:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
        
        # Generate baseline metrics for the original prompt
        baseline_response = self.generator.generate(
            prompt, max_length=self.max_length, temperature=self.temperature
        )
        baseline_metrics = self.evaluator.evaluate(
            prompt, baseline_response, task_type=task_type
        )
        
        # Run the optimization
        optimized_prompt, optimization_history = optimizer.optimize(
            original_prompt=prompt,
            analyzer=self.analyzer,
            generator=self.generator,
            evaluator=self.evaluator,
            task_type=task_type,
            target_audience=target_audience,
            optimization_goals=optimization_goals,
            num_iterations=num_iterations
        )
        
        # Evaluate the optimized prompt
        optimized_response = self.generator.generate(
            optimized_prompt, max_length=self.max_length, temperature=self.temperature
        )
        optimized_metrics = self.evaluator.evaluate(
            optimized_prompt, optimized_response, task_type=task_type
        )
        
        # Calculate improvement metrics
        improvement_metrics = {
            k: optimized_metrics[k] - baseline_metrics[k]
            for k in baseline_metrics.keys() if k in optimized_metrics
        }
        
        # Compile all metrics
        metrics = {
            "baseline": baseline_metrics,
            "optimized": optimized_metrics,
            "improvement": improvement_metrics,
            "optimization_history": optimization_history
        }
        
        logger.info(f"Optimization complete.")
        logger.info(f"Optimized prompt: {optimized_prompt}")
        logger.info(f"Improvement: {improvement_metrics}")
        
        return optimized_prompt, metrics
    
    def compare(
        self,
        original_prompt: str,
        optimized_prompt: str,
        visualize: bool = False,
    ) -> Dict:
        """
        Compare the original and optimized prompts.
        
        Args:
            original_prompt: The original prompt
            optimized_prompt: The optimized prompt
            visualize: Whether to generate visualization
            
        Returns:
            Dictionary with comparison metrics
        """
        # Generate responses for both prompts
        original_response = self.generator.generate(
            original_prompt, max_length=self.max_length, temperature=self.temperature
        )
        optimized_response = self.generator.generate(
            optimized_prompt, max_length=self.max_length, temperature=self.temperature
        )
        
        # Evaluate both
        original_metrics = self.evaluator.evaluate(original_prompt, original_response)
        optimized_metrics = self.evaluator.evaluate(optimized_prompt, optimized_response)
        
        # Calculate differences
        diff_metrics = {
            k: optimized_metrics[k] - original_metrics[k]
            for k in original_metrics.keys() if k in optimized_metrics
        }
        
        comparison = {
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "original_response": original_response,
            "optimized_response": optimized_response,
            "original_metrics": original_metrics,
            "optimized_metrics": optimized_metrics,
            "differences": diff_metrics,
        }
        
        if visualize:
            # Import here to avoid dependency if not used
            from ..utils.visualization import plot_comparison
            comparison["visualization"] = plot_comparison(comparison)
        
        return comparison
    
    def batch_optimize(
        self,
        prompts: List[str],
        task_types: Union[str, List[str]] = "general",
        optimization_goals: List[str] = None,
        num_iterations: int = 5,
    ) -> List[Tuple[str, Dict]]:
        """
        Optimize a batch of prompts.
        
        Args:
            prompts: List of prompts to optimize
            task_types: Either a single task type for all prompts or a list of task types
            optimization_goals: Goals for optimization
            num_iterations: Number of iterations per prompt
            
        Returns:
            List of (optimized_prompt, metrics) tuples
        """
        if isinstance(task_types, str):
            task_types = [task_types] * len(prompts)
            
        if len(task_types) != len(prompts):
            raise ValueError("Length of task_types must match length of prompts")
            
        results = []
        for i, (prompt, task_type) in enumerate(zip(prompts, task_types)):
            logger.info(f"Optimizing prompt {i+1}/{len(prompts)}")
            result = self.optimize(
                prompt=prompt,
                task_type=task_type,
                optimization_goals=optimization_goals,
                num_iterations=num_iterations,
                verbose=False
            )
            results.append(result)
            
        return results