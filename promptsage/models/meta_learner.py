"""
Meta-learning optimizer module for PromptSage
"""

from typing import Dict, List, Tuple, Optional
import random
import numpy as np
import json
import os

class MetaLearningOptimizer:
    """
    Implements meta-learning for prompt optimization.
    Learns optimization patterns across different tasks and prompts.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the meta-learning optimizer.
        
        Args:
            model_path: Path to saved meta-learning model (if available)
        """
        # Load template patterns and transformations
        self.transformation_templates = self._initialize_templates()
        
        # Memory of successful transformations
        self.transformation_memory = {}
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'r') as f:
                    saved_model = json.load(f)
                    self.transformation_memory = saved_model.get('transformation_memory', {})
                    additional_templates = saved_model.get('transformation_templates', [])
                    
                    # Merge with default templates
                    for template in additional_templates:
                        if template not in self.transformation_templates:
                            self.transformation_templates.append(template)
            except Exception as e:
                print(f"Error loading meta-learning model: {e}")
                
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
        Optimize the prompt using meta-learning.
        
        Args:
            original_prompt: The prompt to optimize
            analyzer: PromptAnalyzer instance
            generator: PromptGenerator instance
            evaluator: PromptEvaluator instance
            task_type: Type of task
            target_audience: Target audience
            optimization_goals: Optimization goals
            num_iterations: Number of iterations
            
        Returns:
            Tuple of (best_prompt, optimization_history)
        """
        if optimization_goals is None:
            optimization_goals = ["clarity", "specificity", "relevance"]
        
        # Analyze the original prompt
        prompt_analysis = analyzer.analyze(original_prompt)
        
        # Create a set of candidate transformations based on analysis
        candidate_transformations = self._select_candidate_transformations(
            prompt_analysis, task_type, optimization_goals
        )
        
        current_prompt = original_prompt
        best_prompt = original_prompt
        best_score = 0.0
        
        # Track optimization history
        optimization_history = []
        
        # Generate initial response and evaluation
        initial_response = generator.generate(original_prompt)
        initial_metrics = evaluator.evaluate(
            original_prompt, initial_response, task_type=task_type
        )
        current_score = self._calculate_score(initial_metrics, optimization_goals)
        best_score = current_score
        
        # Optimization loop
        for iteration in range(num_iterations):
            # Apply transformations and evaluate
            candidates = []
            
            for transform_name, transform_fn in candidate_transformations.items():
                # Apply transformation
                transformed_prompt = transform_fn(current_prompt, prompt_analysis)
                
                # Generate response
                response = generator.generate(transformed_prompt)
                
                # Evaluate
                metrics = evaluator.evaluate(
                    transformed_prompt, response, task_type=task_type
                )
                
                # Calculate score
                score = self._calculate_score(metrics, optimization_goals)
                
                candidates.append({
                    "prompt": transformed_prompt,
                    "transform": transform_name,
                    "score": score,
                    "metrics": metrics
                })
            
            # Sort candidates by score (descending)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Record best candidate
            best_candidate = candidates[0]
            
            # Update knowledge base with this result
            self._update_transformation_memory(
                task_type, 
                best_candidate["transform"], 
                best_candidate["score"]
            )
            
            # Check if improvement
            if best_candidate["score"] > best_score:
                best_prompt = best_candidate["prompt"]
                best_score = best_candidate["score"]
                
            # Update current prompt to continue optimization
            current_prompt = best_candidate["prompt"]
            
            # Record history
            optimization_history.append({
                "iteration": iteration,
                "best_prompt": best_prompt,
                "best_score": best_score,
                "candidates": [(c["prompt"], c["transform"], c["score"]) for c in candidates[:3]]
            })
        
        return best_prompt, optimization_history
    
    def _initialize_templates(self) -> Dict:
        """Initialize transformation templates."""
        return {
            "add_specificity": lambda prompt, analysis: self._add_specificity(prompt, analysis),
            "add_context": lambda prompt, analysis: self._add_context(prompt, analysis),
            "restructure_as_question": lambda prompt, analysis: self._restructure_as_question(prompt, analysis),
            "restructure_as_instruction": lambda prompt, analysis: self._restructure_as_instruction(prompt, analysis),
            "add_examples_request": lambda prompt, analysis: self._add_examples_request(prompt, analysis),
            "formalize": lambda prompt, analysis: self._formalize(prompt, analysis),
            "simplify": lambda prompt, analysis: self._simplify(prompt, analysis),
            "add_constraints": lambda prompt, analysis: self._add_constraints(prompt, analysis),
            "add_perspective": lambda prompt, analysis: self._add_perspective(prompt, analysis),
            "add_step_request": lambda prompt, analysis: self._add_step_request(prompt, analysis),
        }
    
    def _select_candidate_transformations(
        self,
        prompt_analysis: Dict,
        task_type: str,
        optimization_goals: List[str]
    ) -> Dict:
        """Select appropriate transformations based on analysis and goals."""
        candidates = {}
        
        # Prioritize transformations based on task type
        task_priorities = {
            "explanation": ["add_step_request", "add_examples_request", "restructure_as_question"],
            "creative": ["add_constraints", "add_perspective", "add_context"],
            "factual": ["add_specificity", "formalize", "add_context"],
            "instruction": ["restructure_as_instruction", "add_step_request", "add_specificity"],
            "coding": ["add_constraints", "add_step_request", "add_specificity"],
            "opinion": ["add_perspective", "restructure_as_question", "add_context"],
            "comparison": ["add_specificity", "add_constraints", "restructure_as_instruction"],
            "definition": ["formalize", "add_context", "add_examples_request"],
            "general": ["add_specificity", "restructure_as_instruction", "add_context"],
        }
        
        # Get priorities for this task type
        priority_transforms = task_priorities.get(task_type, task_priorities["general"])
        
        # Add all priority transformations
        for transform_name in priority_transforms:
            transform_fn = self.transformation_templates.get(transform_name)
            if transform_fn:
                candidates[transform_name] = transform_fn
        
        # Add transformations based on analysis
        if prompt_analysis.get("specificity_score", 0.5) < 0.6:
            candidates["add_specificity"] = self.transformation_templates["add_specificity"]
            
        if prompt_analysis.get("clarity_score", 0.5) < 0.6:
            candidates["simplify"] = self.transformation_templates["simplify"]
        
        # Add more based on optimization goals
        if "creativity" in optimization_goals:
            candidates["add_perspective"] = self.transformation_templates["add_perspective"]
            
        if "clarity" in optimization_goals:
            candidates["restructure_as_instruction"] = self.transformation_templates["restructure_as_instruction"]
        
        # If we have memory of successful transformations for this task type, prioritize them
        task_memory = self.transformation_memory.get(task_type, {})
        if task_memory:
            # Sort by success score
            best_transforms = sorted(
                task_memory.items(), 
                key=lambda x: x[1]["success_score"], 
                reverse=True
            )[:3]
            
            for transform_name, _ in best_transforms:
                transform_fn = self.transformation_templates.get(transform_name)
                if transform_fn:
                    candidates[transform_name] = transform_fn
        
        # If no candidates, add some defaults
        if not candidates:
            for name, fn in list(self.transformation_templates.items())[:3]:
                candidates[name] = fn
        
        return candidates
    
    def _calculate_score(
        self,
        metrics: Dict[str, float],
        optimization_goals: List[str]
    ) -> float:
        """Calculate score based on metrics and optimization goals."""
        score = 0.0
        
        # Map optimization goals to metrics
        goal_metric_map = {
            "clarity": ["coherence_score", "clarity_score"],
            "specificity": ["specificity_score"],
            "relevance": ["relevance_score"],
            "factuality": ["factuality_score"],
            "creativity": ["creativity_score"],
            "fluency": ["perplexity"]  # Lower perplexity is better
        }
        
        # Special handling for perplexity (lower is better)
        if "perplexity" in metrics and "fluency" in optimization_goals:
            if metrics["perplexity"] > 0:
                perplexity_score = 1.0 / (1.0 + metrics["perplexity"])
                score += perplexity_score
            
        # Count matching metrics for each goal
        goal_matches = 0
        for goal in optimization_goals:
            relevant_metrics = goal_metric_map.get(goal, [])
            for metric in relevant_metrics:
                if metric in metrics and metric != "perplexity":  # Already handled perplexity
                    score += metrics[metric]
                    goal_matches += 1
        
        # Normalize
        normalizer = len(optimization_goals) + goal_matches
        if normalizer > 0:
            score /= normalizer
        
        return score
    
    def _update_transformation_memory(
        self,
        task_type: str,
        transform_name: str,
        score: float
    ) -> None:
        """Update memory with transformation results."""
        # Initialize task type if not exists
        if task_type not in self.transformation_memory:
            self.transformation_memory[task_type] = {}
            
        # Initialize transformation if not exists
        if transform_name not in self.transformation_memory[task_type]:
            self.transformation_memory[task_type][transform_name] = {
                "success_count": 0,
                "total_count": 0,
                "success_score": 0.0
            }
            
        # Update counts
        self.transformation_memory[task_type][transform_name]["total_count"] += 1
        
        # If score is above threshold, count as success
        if score > 0.6:  # Success threshold
            self.transformation_memory[task_type][transform_name]["success_count"] += 1
            
        # Update success score (higher score = more successful)
        memory = self.transformation_memory[task_type][transform_name]
        memory["success_score"] = (
            memory["success_count"] / memory["total_count"] * score
        )
    
    def save_model(self, path: str) -> None:
        """Save the meta-learning model to disk."""
        model_data = {
            "transformation_memory": self.transformation_memory,
            "transformation_templates": list(self.transformation_templates.keys())
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    # Transformation functions
    def _add_specificity(self, prompt: str, analysis: Dict) -> str:
        """Add specificity to the prompt."""
        specificity_phrases = [
            "specifically ",
            "in detail ",
            "with clear examples ",
            "with precise steps ",
            "with concrete instances "
        ]
        
        # Choose a phrase
        phrase = random.choice(specificity_phrases)
        
        # Add phrase at appropriate position
        if "?" in prompt:
            # If it's a question, insert before the question mark
            parts = prompt.rsplit("?", 1)
            return f"{parts[0]} {phrase}?{parts[1] if len(parts) > 1 else ''}"
        else:
            # Otherwise add to the end
            prompt = prompt.rstrip()
            if prompt.endswith((".", "!", ":")):
                return f"{prompt} Please be {phrase}in your response."
            else:
                return f"{prompt}. Please be {phrase}in your response."
    
    def _add_context(self, prompt: str, analysis: Dict) -> str:
        """Add contextual information to the prompt."""
        context_prefixes = [
            "For a beginner audience, ",
            "In simple terms, ",
            "From an expert perspective, ",
            "In the context of recent developments, ",
            "As if explaining to someone unfamiliar with the topic, "
        ]
        
        prefix = random.choice(context_prefixes)
        
        # Add context prefix
        return f"{prefix}{prompt}"
    
    def _restructure_as_question(self, prompt: str, analysis: Dict) -> str:
        """Restructure the prompt as a question if it isn't already."""
        if prompt.strip().endswith("?"):
            return prompt  # Already a question
            
        # Convert to question
        if prompt.lower().startswith(("what", "why", "how", "when", "where", "who")):
            # Might be a question without question mark
            return f"{prompt}?"
            
        # Transform statement to question
        prompt = prompt.rstrip(".!:;, ")
        question_starters = [
            f"Can you explain {prompt}?",
            f"What are the key aspects of {prompt}?",
            f"How would you describe {prompt}?",
            f"Could you elaborate on {prompt}?",
            f"What should I know about {prompt}?"
        ]
        
        return random.choice(question_starters)
    
    def _restructure_as_instruction(self, prompt: str, analysis: Dict) -> str:
        """Restructure the prompt as a clear instruction."""
        instruction_verbs = ["explain", "describe", "outline", "detail", "elaborate on"]
        verb = random.choice(instruction_verbs)
        
        # Remove trailing punctuation
        prompt = prompt.rstrip(".!?:;, ")
        
        # Check if it already starts with an instruction verb
        for v in instruction_verbs:
            if prompt.lower().startswith(v):
                return prompt  # Already an instruction
                
        # If it's a question, convert to instruction
        if prompt.strip().endswith("?"):
            # Convert question to instruction
            prompt = prompt.rstrip("?")
            # Remove question words
            for qword in ["what is", "what are", "how does", "how do", "why is", "why are"]:
                if prompt.lower().startswith(qword):
                    prompt = prompt[len(qword):].strip()
                    break
            
        return f"{verb} {prompt}."
    
    def _add_examples_request(self, prompt: str, analysis: Dict) -> str:
        """Add a request for examples."""
        example_requests = [
            " Include specific examples.",
            " Provide real-world examples in your answer.",
            " Use concrete examples to illustrate.",
            " Support your explanation with examples.",
            " Give at least 3 examples."
        ]
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{random.choice(example_requests)}"
        else:
            return f"{prompt}.{random.choice(example_requests)}"
    
    def _formalize(self, prompt: str, analysis: Dict) -> str:
        """Make the prompt more formal."""
        # Replace informal words with formal ones
        informal_to_formal = {
            "get": "obtain",
            "show": "demonstrate",
            "tell": "explain",
            "find out": "determine",
            "look at": "examine",
            "lots of": "numerous",
            "big": "significant",
            "small": "minimal",
            "good": "beneficial",
            "bad": "detrimental"
        }
        
        result = prompt
        for informal, formal in informal_to_formal.items():
            result = result.replace(informal, formal)
            
        # Add formal framing
        formal_frames = [
            "I would appreciate a detailed explanation of ",
            "Please provide a comprehensive analysis of ",
            "I request a thorough description of ",
            "Kindly elaborate on the topic of ",
            "I am seeking a scholarly explanation of "
        ]
        
        # Check if we should add formal framing
        if not any(result.startswith(frame.split()[0]) for frame in formal_frames):
            # Only add framing if not already formal
            frame = random.choice(formal_frames)
            result = f"{frame}{result}"
            
        return result
    
    def _simplify(self, prompt: str, analysis: Dict) -> str:
        """Simplify complex prompts."""
        # If the prompt is long, try to simplify
        if len(prompt.split()) > 20:
            # Extract main topic (naive approach - take first and last few words)
            words = prompt.split()
            simplified = " ".join(words[:5] + ["..."] + words[-5:])
            return f"Explain this concisely: {simplified}"
            
        # If sentences are long, break them up
        sentences = [s.strip() for s in prompt.split(".") if s.strip()]
        if any(len(s.split()) > 15 for s in sentences):
            return "Break this down in simple terms: " + prompt
            
        # Otherwise, just request simplicity
        return f"{prompt.rstrip()} Explain in simple, clear terms."
    
    def _add_constraints(self, prompt: str, analysis: Dict) -> str:
        """Add constraints to guide the response format."""
        constraints = [
            " Limit your answer to 3-5 key points.",
            " Structure your response with bullet points.",
            " Keep your explanation under 100 words.",
            " Organize your answer in a step-by-step format.",
            " Focus only on the most essential aspects."
        ]
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{random.choice(constraints)}"
        else:
            return f"{prompt}.{random.choice(constraints)}"
    
    def _add_perspective(self, prompt: str, analysis: Dict) -> str:
        """Add a perspective angle to the prompt."""
        perspectives = [
            "from different viewpoints",
            "considering both advantages and disadvantages",
            "from a historical perspective",
            "from a future-oriented perspective",
            "comparing different approaches",
            "considering ethical implications"
        ]
        
        perspective = random.choice(perspectives)
        prompt = prompt.rstrip()
        
        if prompt.endswith((".", "!", "?")):
            return f"{prompt} Analyze this {perspective}."
        else:
            return f"{prompt}. Analyze this {perspective}."
    
    def _add_step_request(self, prompt: str, analysis: Dict) -> str:
        """Add a request for step-by-step explanation."""
        step_requests = [
            " Explain this in step-by-step detail.",
            " Break this down into sequential steps.",
            " Provide a step-by-step guide.",
            " Walk me through this process systematically.",
            " Outline the procedure in ordered steps."
        ]
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{random.choice(step_requests)}"
        else:
            return f"{prompt}.{random.choice(step_requests)}"