"""
Evolutionary optimization module for PromptSage
"""

import random
import re
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

class EvolutionaryOptimizer:
    """
    Implements genetic algorithm for prompt optimization.
    Treats prompt components as genes that evolve over generations.
    """
    
    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elitism_count: int = 2,
        max_generations: int = 10,
    ):
        """
        Initialize the evolutionary optimizer.
        
        Args:
            population_size: Size of prompt population
            mutation_rate: Probability of mutating a prompt component
            crossover_rate: Probability of performing crossover
            elitism_count: Number of top prompts to preserve unchanged
            max_generations: Maximum number of generations to evolve
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.max_generations = max_generations
        
    def optimize(
        self,
        original_prompt: str,
        analyzer: any,  # PromptAnalyzer
        generator: any,  # PromptGenerator
        evaluator: any,  # PromptEvaluator
        task_type: str = "general",
        target_audience: str = "general",
        optimization_goals: List[str] = None,
        num_iterations: int = 10,
    ) -> Tuple[str, List[Dict]]:
        """
        Evolve the prompt using genetic algorithm.
        
        Args:
            original_prompt: The prompt to optimize
            analyzer: PromptAnalyzer instance
            generator: PromptGenerator instance
            evaluator: PromptEvaluator instance
            task_type: Type of task
            target_audience: Target audience
            optimization_goals: Optimization goals
            num_iterations: Number of iterations/generations
            
        Returns:
            Tuple of (best_prompt, optimization_history)
        """
        if optimization_goals is None:
            optimization_goals = ["clarity", "specificity", "relevance"]
            
        # Limit generations to specified iterations
        generations = min(self.max_generations, num_iterations)
        
        # Initialize population
        population = self._initialize_population(original_prompt)
        
        # Evaluation function that combines all goals
        fitness_fn = lambda prompt: self._calculate_fitness(
            prompt, generator, evaluator, task_type, optimization_goals
        )
        
        # Track history
        optimization_history = []
        
        # Evolve for specified generations
        for generation in range(generations):
            # Evaluate all prompts
            fitness_scores = [fitness_fn(prompt) for prompt in population]
            
            # Track best prompt and score
            best_idx = np.argmax(fitness_scores)
            best_prompt = population[best_idx]
            best_score = fitness_scores[best_idx]
            
            # Record history
            optimization_history.append({
                "generation": generation,
                "best_prompt": best_prompt,
                "best_score": best_score,
                "population_size": len(population),
                "avg_score": np.mean(fitness_scores),
            })
            
            # Create next generation
            population = self._create_next_generation(
                population, fitness_scores, analyzer
            )
        
        # Return best prompt from final generation
        final_fitness_scores = [fitness_fn(prompt) for prompt in population]
        best_idx = np.argmax(final_fitness_scores)
        best_prompt = population[best_idx]
        
        return best_prompt, optimization_history
    
    def _initialize_population(self, original_prompt: str) -> List[str]:
        """Create initial population from original prompt."""
        population = [original_prompt]
        
        # Create variations of the original prompt
        for _ in range(self.population_size - 1):
            # Apply random transformations to create diverse initial population
            variant = self._create_prompt_variant(original_prompt)
            population.append(variant)
            
        return population
    
    def _create_prompt_variant(self, prompt: str) -> str:
        """Create a variant of a prompt with random transformations."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', prompt)
        
        # Apply random transformations
        variant = prompt
        
        # Randomly select a transformation
        transform_type = random.choice([
            "add_specificity",
            "change_tone",
            "restructure",
            "add_context",
            "simplify"
        ])
        
        if transform_type == "add_specificity":
            # Add specificity phrases
            specificity_phrases = [
                " specifically ",
                " in particular ",
                " with detailed examples ",
                " with precise steps ",
                " with clear reasoning "
            ]
            insertion_point = random.randint(0, len(variant) - 1)
            phrase = random.choice(specificity_phrases)
            variant = variant[:insertion_point] + phrase + variant[insertion_point:]
            
        elif transform_type == "change_tone":
            # Change tone words
            formal_phrases = [
                ("tell me", "please elaborate on"),
                ("show", "demonstrate"),
                ("good", "excellent"),
                ("bad", "problematic"),
                ("big", "substantial"),
            ]
            for original, replacement in formal_phrases:
                if random.random() < 0.3 and original in variant.lower():
                    variant = re.sub(r'\b' + re.escape(original) + r'\b', 
                                   replacement, 
                                   variant, 
                                   flags=re.IGNORECASE)
                    
        elif transform_type == "restructure":
            # Restructure if multiple sentences
            if len(sentences) > 1:
                random.shuffle(sentences)
                variant = " ".join(sentences)
                
        elif transform_type == "add_context":
            # Add context prefix
            context_prefixes = [
                "For a beginner audience, ",
                "As if explaining to a five-year-old, ",
                "From an expert perspective, ",
                "In the context of modern developments, ",
                "Considering historical precedents, "
            ]
            prefix = random.choice(context_prefixes)
            variant = prefix + variant
            
        elif transform_type == "simplify":
            # Simplify by truncating if long
            if len(variant) > 50:
                end_pos = random.randint(int(len(variant) * 0.7), len(variant))
                variant = variant[:end_pos]
                if not variant.endswith((".", "?", "!")):
                    variant += "."
        
        return variant
    
    def _create_next_generation(
        self, 
        population: List[str], 
        fitness_scores: List[float],
        analyzer: any  # PromptAnalyzer
    ) -> List[str]:
        """Create the next generation through selection, crossover, and mutation."""
        # Normalize fitness scores for selection probability
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            selection_probs = [1.0 / len(population)] * len(population)
        else:
            selection_probs = [score / total_fitness for score in fitness_scores]
            
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        
        # New population with elitism (keeping best prompts)
        new_population = sorted_population[:self.elitism_count]
        
        # Fill the rest through selection, crossover, mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(population) >= 2:
                # Select parents
                parent1 = random.choices(population, weights=selection_probs)[0]
                parent2 = random.choices(population, weights=selection_probs)[0]
                
                # Create child through crossover
                child = self._crossover(parent1, parent2)
                
                # Mutate
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, analyzer)
                    
                new_population.append(child)
            else:
                # Just select and mutate
                parent = random.choices(population, weights=selection_probs)[0]
                if random.random() < self.mutation_rate:
                    parent = self._mutate(parent, analyzer)
                new_population.append(parent)
        
        return new_population
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """Create a new prompt by combining parts of two parent prompts."""
        # Simple crossover: take beginning from one parent and end from other
        if random.random() < 0.5:
            # Sentence-level crossover if possible
            sentences1 = re.split(r'(?<=[.!?])\s+', parent1)
            sentences2 = re.split(r'(?<=[.!?])\s+', parent2)
            
            if len(sentences1) > 1 and len(sentences2) > 1:
                # Random crossover points
                crossover_point1 = random.randint(1, len(sentences1) - 1)
                crossover_point2 = random.randint(1, len(sentences2) - 1)
                
                # Create child
                child = " ".join(sentences1[:crossover_point1] + sentences2[crossover_point2:])
                return child
        
        # Word-level crossover (fallback)
        words1 = parent1.split()
        words2 = parent2.split()
        
        crossover_point1 = random.randint(1, len(words1) - 1)
        crossover_point2 = random.randint(1, len(words2) - 1)
        
        child = " ".join(words1[:crossover_point1] + words2[crossover_point2:])
        
        return child
    
    def _mutate(self, prompt: str, analyzer: any) -> str:
        """Mutate a prompt by making small changes."""
        # Analyze to guide mutation
        analysis = analyzer.analyze(prompt)
        
        # Choose mutation type based on analysis
        if analysis["clarity_score"] < 0.6:
            return self._mutate_for_clarity(prompt)
        elif analysis["specificity_score"] < 0.6:
            return self._mutate_for_specificity(prompt)
        else:
            # Random mutation
            mutation_types = [
                self._mutate_for_clarity,
                self._mutate_for_specificity,
                self._mutate_sentence_structure,
                self._mutate_add_examples
            ]
            mutation_fn = random.choice(mutation_types)
            return mutation_fn(prompt)
    
    def _mutate_add_examples(self, prompt: str) -> str:
        """Mutate by requesting examples."""
        example_phrases = [
            ". Include specific examples.",
            ". Provide concrete examples.",
            ". Use real-world examples.",
            ". Illustrate with examples."
        ]
    
        # Avoid adding if already there
        if "example" in prompt.lower():
            return prompt
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{random.choice(example_phrases)}"
        else:
            return f"{prompt}.{random.choice(example_phrases)}"
    
    
    def _mutate_for_clarity(self, prompt: str) -> str:
        """Mutate to improve clarity."""
        # Replace vague words with more specific ones
        vague_word_replacements = {
            "thing": ["item", "element", "component", "object"],
            "stuff": ["material", "content", "substance", "items"],
            "good": ["excellent", "beneficial", "valuable", "positive"],
            "bad": ["problematic", "negative", "harmful", "inferior"],
            "nice": ["pleasant", "appealing", "satisfying", "delightful"],
            "a lot": ["significantly", "substantially", "considerably"]
        }
        
        result = prompt
        
        # Replace vague words
        for vague_word, replacements in vague_word_replacements.items():
            if vague_word in result.lower():
                replacement = random.choice(replacements)
                result = re.sub(r'\b' + re.escape(vague_word) + r'\b', 
                               replacement, 
                               result, 
                               flags=re.IGNORECASE,
                               count=1)
        
        return result
    
    def _mutate_for_specificity(self, prompt: str) -> str:
        """Mutate to improve specificity."""
        specificity_phrases = [
             " specifically ",
             " in particular ",
             " for example ",
             " to be precise ",
             " namely "
         ]
    
        # Split into words to respect word boundaries
        words = prompt.split()
    
        if len(words) < 2:
            return prompt + specificity_phrases[0].strip()
    
        # Insert at word boundary
        insert_idx = random.randint(1, len(words) - 1)
        phrase = random.choice(specificity_phrases)
    
        result = " ".join(words[:insert_idx]) + phrase + " ".join(words[insert_idx:])
        return result
    
    def _mutate_sentence_structure(self, prompt: str) -> str:
        """Mutate sentence structure."""
        # Change sentence structure if it's a question
        if prompt.strip().endswith("?"):
            prefixes = [
                "I'm interested in understanding ",
                "Could you elaborate on ",
                "I'd like to know more about ",
                "Please explain ",
                "Help me understand "
            ]
            
            # Remove question mark and add prefix
            result = random.choice(prefixes) + prompt.rstrip("?") + "."
            return result
            
        # Change statement to more directed request
        instruction_verbs = ["explain", "describe", "outline", "elaborate on", "analyze"]
        
        if not any(verb in prompt.lower() for verb in instruction_verbs):
            verb = random.choice(instruction_verbs)
            result = f"{verb} {prompt}"
            return result
            
        return prompt  # No change if conditions not met
    
    def _add_examples_request(self, prompt: str) -> str:
        """Mutate by requesting examples."""
        example_phrases = [
            ". Include specific examples.",
            ". Provide concrete examples.",
            ". Use real-world examples.",
            ". Illustrate with examples."
        ]
    
        # Respect sentence endings
        prompt = prompt.rstrip()
        if any(prompt.endswith(end) for end in [".", "!", "?"]):
            return f"{prompt}{random.choice(example_phrases)}"
        else:
            return f"{prompt}.{random.choice(example_phrases)}"
    
    def _calculate_fitness(
        self,
        prompt: str,
        generator: any,  # PromptGenerator
        evaluator: any,  # PromptEvaluator
        task_type: str,
        optimization_goals: List[str]
    ) -> float:
        """Calculate fitness score for a prompt based on optimization goals."""
        # Generate response
        response = generator.generate(prompt, max_length=100, temperature=0.7)
        
        # Evaluate
        metrics = evaluator.evaluate(prompt, response, task_type=task_type)
        
        # Calculate weighted score based on optimization goals
        score = 0.0
        
        # Define weights for different metrics based on goals
        goal_metric_map = {
            "clarity": ["coherence_score", "clarity_score"],
            "specificity": ["specificity_score"],
            "relevance": ["relevance_score"],
            "factuality": ["factuality_score"],
            "creativity": ["creativity_score"],
            "fluency": ["perplexity"]  # Lower perplexity is better
        }
        
        # Count goals for weight normalization
        metric_weights = {}
        
        for goal in optimization_goals:
            relevant_metrics = goal_metric_map.get(goal, [])
            for metric in relevant_metrics:
                if metric in metrics:
                    metric_weights[metric] = metric_weights.get(metric, 0) + 1
        
        # Special handling for perplexity (lower is better)
        if "perplexity" in metrics and "perplexity" in metric_weights:
            # Invert and normalize perplexity (higher score = better)
            if metrics["perplexity"] > 0:
                perplexity_score = 1.0 / (1.0 + metrics["perplexity"])
            else:
                perplexity_score = 0.5  # Default if undefined
                
            score += perplexity_score * metric_weights["perplexity"]
            
            # Remove from regular processing
            metrics.pop("perplexity")
            metric_weights.pop("perplexity")
        
        # Add scores for all other metrics
        for metric, weight in metric_weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        
        # Normalize score
        total_weight = sum(metric_weights.values()) + (1 if "perplexity" in metrics else 0)
        if total_weight > 0:
            score /= total_weight
        
        return score