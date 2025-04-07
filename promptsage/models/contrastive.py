"""
Contrastive learning module for PromptSage
"""

from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np

class ContrastiveOptimizer:
    """
    Implements contrastive learning for prompt optimization.
    Learns from paired examples of effective and ineffective prompts.
    """
    
    def __init__(
        self, 
        contrastive_pairs: Optional[List[Tuple[str, str, float]]] = None,
        learning_rate: float = 0.1,
    ):
        """
        Initialize the contrastive optimizer.
        
        Args:
            contrastive_pairs: Optional initial pairs of (good_prompt, bad_prompt, score_diff)
            learning_rate: Learning rate for updating patterns
        """
        # Initialize contrastive pairs
        self.contrastive_pairs = contrastive_pairs or []
        self.learning_rate = learning_rate
        
        # Learned pattern weights
        self.pattern_weights = self._initialize_pattern_weights()
    
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
        Optimize the prompt using contrastive learning.
        
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
            
        # Initialize best prompt and history
        best_prompt = original_prompt
        current_prompt = original_prompt
        optimization_history = []
        
        # Generate initial response and metrics
        initial_response = generator.generate(original_prompt)
        initial_metrics = evaluator.evaluate(
            original_prompt, initial_response, task_type=task_type
        )
        initial_score = self._calculate_score(initial_metrics, optimization_goals)
        best_score = initial_score
        
        for iteration in range(num_iterations):
            # Generate contrastive pairs for learning
            positive_variants = []
            negative_variants = []
            
            # Generate multiple variants
            variants = self._generate_prompt_variants(
                current_prompt, analyzer, task_type
            )
            
            # Evaluate all variants
            variant_scores = []
            for variant in variants:
                response = generator.generate(variant)
                metrics = evaluator.evaluate(variant, response, task_type=task_type)
                score = self._calculate_score(metrics, optimization_goals)
                variant_scores.append((variant, score))
            
            # Sort by score
            variant_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select positive and negative examples
            if len(variant_scores) >= 4:
                # Take top 2 as positive, bottom 2 as negative
                positive_variants = [variant_scores[0][0], variant_scores[1][0]]
                negative_variants = [variant_scores[-1][0], variant_scores[-2][0]]
                
                # Calculate score differences for learning
                pos_score_diff = variant_scores[0][1] - initial_score
                neg_score_diff = initial_score - variant_scores[-1][1]
                
                # Add to contrastive pairs if significant difference
                if pos_score_diff > 0.05:
                    self.contrastive_pairs.append((
                        positive_variants[0], original_prompt, pos_score_diff
                    ))
                if neg_score_diff > 0.05:
                    self.contrastive_pairs.append((
                        original_prompt, negative_variants[0], neg_score_diff
                    ))
                    
                # Learn from new pairs
                self._learn_from_contrastive_pairs()
            
            # Select the best variant as the new current prompt
            if variant_scores and variant_scores[0][1] > best_score:
                best_prompt = variant_scores[0][0]
                best_score = variant_scores[0][1]
                current_prompt = best_prompt
            
            # Record history
            optimization_history.append({
                "iteration": iteration,
                "best_prompt": best_prompt,
                "best_score": best_score,
                "variants_evaluated": len(variants),
                "top_variant_score": variant_scores[0][1] if variant_scores else None,
                "pattern_weights": {k: round(v, 3) for k, v in self.pattern_weights.items()},
            })
        
        return best_prompt, optimization_history
    
    def _initialize_pattern_weights(self) -> Dict[str, float]:
        """Initialize weights for different prompt patterns."""
        return {
            # Structural patterns
            "starts_with_instruction_verb": 0.6,
            "ends_with_question": 0.5,
            "includes_context_setting": 0.6,
            "includes_examples_request": 0.6,
            "includes_step_request": 0.7,
            
            # Content patterns
            "includes_specificity_markers": 0.5,
            "includes_constraints": 0.5,
            "includes_perspective_request": 0.4,
            
            # Complexity patterns
            "simple_sentence_structure": 0.5,
            "moderate_length": 0.6,
            "formal_language": 0.4,
            
            # Target patterns
            "addresses_audience": 0.5,
            "explains_purpose": 0.6,
            "sets_format_expectations": 0.5,
        }
    
    def _generate_prompt_variants(
        self,
        prompt: str,
        analyzer: Any,
        task_type: str
    ) -> List[str]:
        """Generate multiple prompt variants for contrastive learning."""
        variants = []
        analysis = analyzer.analyze(prompt)
        
        # Apply transformations based on learned pattern weights
        transformations = [
            self._add_instruction_verb,
            self._add_question_form,
            self._add_context_setting,
            self._add_examples_request,
            self._add_step_request,
            self._add_specificity_markers,
            self._add_constraints,
            self._add_perspective_request,
            self._simplify_structure,
            self._adjust_length,
            self._formalize_language,
            self._address_audience,
            self._explain_purpose,
            self._set_format_expectations,
        ]
        
        # Weight transformations based on learned weights
        transformation_weights = {
            self._add_instruction_verb: self.pattern_weights["starts_with_instruction_verb"],
            self._add_question_form: self.pattern_weights["ends_with_question"],
            self._add_context_setting: self.pattern_weights["includes_context_setting"],
            self._add_examples_request: self.pattern_weights["includes_examples_request"],
            self._add_step_request: self.pattern_weights["includes_step_request"],
            self._add_specificity_markers: self.pattern_weights["includes_specificity_markers"],
            self._add_constraints: self.pattern_weights["includes_constraints"],
            self._add_perspective_request: self.pattern_weights["includes_perspective_request"],
            self._simplify_structure: self.pattern_weights["simple_sentence_structure"],
            self._adjust_length: self.pattern_weights["moderate_length"],
            self._formalize_language: self.pattern_weights["formal_language"],
            self._address_audience: self.pattern_weights["addresses_audience"],
            self._explain_purpose: self.pattern_weights["explains_purpose"],
            self._set_format_expectations: self.pattern_weights["sets_format_expectations"],
        }
        
        # Apply individual transformations
        for transform in transformations:
            if random.random() < transformation_weights[transform]:
                try:
                    variant = transform(prompt, analysis, task_type)
                    if variant != prompt:
                        variants.append(variant)
                except Exception as e:
                    # Skip errored transformations
                    pass
        
        # Apply composite transformations (combinations of 2 transformations)
        if len(transformations) >= 2:
            for _ in range(3):  # Generate 3 composite variants
                # Select two random transformations
                selected = random.sample(transformations, 2)
                try:
                    # Apply sequentially
                    intermediate = selected[0](prompt, analysis, task_type)
                    # Re-analyze the intermediate result
                    intermediate_analysis = analyzer.analyze(intermediate)
                    final = selected[1](intermediate, intermediate_analysis, task_type)
                    if final != prompt and final not in variants:
                        variants.append(final)
                except Exception as e:
                    # Skip errored transformations
                    pass
        
        # Always include the original prompt
        variants.append(prompt)
        
        return variants
    
    def _learn_from_contrastive_pairs(self) -> None:
        """Update pattern weights based on contrastive examples."""
        if not self.contrastive_pairs:
            return
            
        # For each pattern, calculate its presence in positive vs negative examples
        pattern_effectiveness = {pattern: 0.0 for pattern in self.pattern_weights}
        
        for good_prompt, bad_prompt, score_diff in self.contrastive_pairs:
            # Check pattern presence
            good_patterns = self._extract_prompt_patterns(good_prompt)
            bad_patterns = self._extract_prompt_patterns(bad_prompt)
            
            # Update effectiveness
            for pattern in pattern_effectiveness:
                if pattern in good_patterns and pattern not in bad_patterns:
                    # Pattern in good but not bad - positive signal
                    pattern_effectiveness[pattern] += score_diff
                elif pattern in bad_patterns and pattern not in good_patterns:
                    # Pattern in bad but not good - negative signal
                    pattern_effectiveness[pattern] -= score_diff
        
        # Update weights
        for pattern, effectiveness in pattern_effectiveness.items():
            # Apply learning rate and bound weights between 0.1 and 0.9
            self.pattern_weights[pattern] = max(0.1, min(0.9, 
                self.pattern_weights[pattern] + (self.learning_rate * effectiveness)
            ))
    
    def _extract_prompt_patterns(self, prompt: str) -> List[str]:
        """Extract patterns present in a prompt."""
        patterns = []
        
        # Check for each pattern
        # Structural patterns
        if any(prompt.lower().startswith(verb) for verb in ["explain", "describe", "outline", "detail", "elaborate", "analyze"]):
            patterns.append("starts_with_instruction_verb")
            
        if prompt.strip().endswith("?"):
            patterns.append("ends_with_question")
            
        if any(marker in prompt.lower() for marker in ["context", "background", "setting", "scenario", "situation"]):
            patterns.append("includes_context_setting")
            
        if any(marker in prompt.lower() for marker in ["example", "illustration", "instance", "case"]):
            patterns.append("includes_examples_request")
            
        if any(marker in prompt.lower() for marker in ["step", "procedure", "process", "sequence", "stages"]):
            patterns.append("includes_step_request")
            
        # Content patterns
        if any(marker in prompt.lower() for marker in ["specifically", "particular", "exact", "precise", "concrete", "detailed"]):
            patterns.append("includes_specificity_markers")
            
        if any(marker in prompt.lower() for marker in ["limit", "constraint", "restrict", "only", "just", "maximum", "minimum"]):
            patterns.append("includes_constraints")
            
        if any(marker in prompt.lower() for marker in ["perspective", "viewpoint", "angle", "standpoint", "position", "opinion"]):
            patterns.append("includes_perspective_request")
            
        # Complexity patterns
        if len(prompt) < 150 and max(len(s.split()) for s in prompt.split(".")) < 20:
            patterns.append("simple_sentence_structure")
            
        if 50 < len(prompt) < 200:
            patterns.append("moderate_length")
            
        formal_markers = ["would", "could", "please", "kindly", "appreciate", "formal", "professional"]
        if any(marker in prompt.lower() for marker in formal_markers):
            patterns.append("formal_language")
            
        # Target patterns
        audience_markers = ["beginner", "expert", "student", "professional", "layperson", "audience", "reader"]
        if any(marker in prompt.lower() for marker in audience_markers):
            patterns.append("addresses_audience")
            
        purpose_markers = ["goal", "aim", "purpose", "objective", "intention", "trying to", "want to"]
        if any(marker in prompt.lower() for marker in purpose_markers):
            patterns.append("explains_purpose")
            
        format_markers = ["format", "structure", "organize", "bullet", "paragraph", "concise", "brief", "detailed"]
        if any(marker in prompt.lower() for marker in format_markers):
            patterns.append("sets_format_expectations")
            
        return patterns
    
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
    
    # Pattern transformation functions
    def _add_instruction_verb(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Add an instruction verb to the start of the prompt if not present."""
        instruction_verbs = {
            "explanation": ["Explain", "Describe", "Clarify"],
            "comparison": ["Compare", "Contrast", "Differentiate between"],
            "opinion": ["Share your thoughts on", "Give your opinion about", "Evaluate"],
            "instruction": ["Instruct how to", "Guide through", "Show the steps to"],
            "creative": ["Create", "Write", "Compose", "Design"],
            "factual": ["Define", "Provide facts about", "List key information about"],
            "summarization": ["Summarize", "Provide a summary of", "Give an overview of"],
            "analysis": ["Analyze", "Examine", "Investigate"],
            "coding": ["Code", "Implement", "Program", "Write a function to"],
            "general": ["Explain", "Describe", "Detail"]
        }
        
        verbs = instruction_verbs.get(task_type, instruction_verbs["general"])
        
        # Check if already starts with instruction verb
        if any(prompt.lower().startswith(verb.lower()) for verb in [v for vs in instruction_verbs.values() for v in vs]):
            return prompt
            
        # Add instruction verb
        verb = random.choice(verbs)
        
        if prompt.strip().endswith("?"):
            # Convert question to instruction
            question = prompt.strip().rstrip("?")
            return f"{verb} {question}."
        else:
            return f"{verb} {prompt}"
    
    def _add_question_form(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Convert prompt to question form if not already a question."""
        if prompt.strip().endswith("?"):
            return prompt
            
        # Only applicable to certain task types
        applicable_tasks = ["explanation", "opinion", "factual", "comparison", "general"]
        if task_type not in applicable_tasks:
            return prompt
            
        # Question starters by task type
        question_starters = {
            "explanation": ["How would you explain", "Can you describe", "What is"],
            "opinion": ["What do you think about", "How do you feel about", "What's your take on"],
            "factual": ["What are the facts about", "What's known about", "Can you provide information on"],
            "comparison": ["How do you compare", "What are the differences between", "How would you contrast"],
            "general": ["Can you explain", "What can you tell me about", "How would you describe"]
        }
        
        starters = question_starters.get(task_type, question_starters["general"])
        starter = random.choice(starters)
        
        # Remove instruction verbs if present
        instruction_verbs = ["explain", "describe", "elaborate", "detail", "analyze"]
        for verb in instruction_verbs:
            if prompt.lower().startswith(verb):
                prompt = prompt[len(verb):].strip()
                break
                
        return f"{starter} {prompt}?"
    
    def _add_context_setting(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Add context setting to the prompt based on task type."""
        if "includes_context_setting" in self._extract_prompt_patterns(prompt):
            return prompt
            
        # Context settings by task type
        context_settings = {
            "explanation": [
                "For someone new to this topic, ", 
                "As if explaining to a beginner, ",
                "In educational context, "
            ],
            "creative": [
                "In a creative writing context, ",
                "For a fictional scenario, ",
                "With artistic freedom, "
            ],
            "factual": [
                "From a scientific perspective, ",
                "Based on established research, ",
                "According to current understanding, "
            ],
            "coding": [
                "For a production environment, ",
                "As efficient code, ",
                "Following best practices, "
            ],
            "general": [
                "In general terms, ",
                "From a broad perspective, ",
                "In the context of common understanding, "
            ]
        }
        
        settings = context_settings.get(task_type, context_settings["general"])
        setting = random.choice(settings)
        
        return f"{setting}{prompt}"
    
    def _add_examples_request(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Request examples in the prompt."""
        if "includes_examples_request" in self._extract_prompt_patterns(prompt):
            return prompt
            
        if task_type not in ["explanation", "instruction", "factual", "comparison", "general"]:
            return prompt
            
        # Example requests by task type
        example_requests = {
            "explanation": [
                " Include concrete examples.",
                " Provide real-world instances.",
                " Use illustrative examples."
            ],
            "instruction": [
                " Show with specific examples.",
                " Demonstrate with practical examples.",
                " Include example use cases."
            ],
            "factual": [
                " Cite specific instances.",
                " Provide evidence with examples.",
                " Include representative examples."
            ],
            "comparison": [
                " Contrast with clear examples.",
                " Use comparative examples.",
                " Illustrate differences with examples."
            ],
            "general": [
                " Include examples.",
                " Provide some examples.",
                " Use examples to clarify."
            ]
        }
        
        requests = example_requests.get(task_type, example_requests["general"])
        request = random.choice(requests)
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{request}"
        else:
            return f"{prompt}.{request}"
    
    def _add_step_request(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Request step-by-step explanation in the prompt."""
        if "includes_step_request" in self._extract_prompt_patterns(prompt):
            return prompt
            
        if task_type not in ["explanation", "instruction", "analysis", "coding", "general"]:
            return prompt
            
        # Step requests by task type
        step_requests = {
            "explanation": [
                " Break this down into steps.",
                " Explain this step-by-step.",
                " Walk through this sequentially."
            ],
            "instruction": [
                " Provide sequential instructions.",
                " Detail the steps involved.",
                " Give a step-by-step guide."
            ],
            "analysis": [
                " Analyze this in sequential stages.",
                " Break this analysis into clear steps.",
                " Provide a structured analysis with steps."
            ],
            "coding": [
                " Show the implementation steps.",
                " Detail the algorithm steps.",
                " Provide a step-by-step coding solution."
            ],
            "general": [
                " Explain in steps.",
                " Provide a step-by-step approach.",
                " Break this down into stages."
            ]
        }
        
        requests = step_requests.get(task_type, step_requests["general"])
        request = random.choice(requests)
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{request}"
        else:
            return f"{prompt}.{request}"
    
    def _add_specificity_markers(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Add specificity markers to the prompt."""
        if "includes_specificity_markers" in self._extract_prompt_patterns(prompt):
            return prompt
            
        # Specificity markers by task type
        specificity_markers = {
            "explanation": [
                " Be specific and detailed.",
                " Include precise information.",
                " Provide specifics rather than generalities."
            ],
            "factual": [
                " Include specific dates, numbers, and facts.",
                " Provide precise factual details.",
                " Be exact and specific in your response."
            ],
            "coding": [
                " Include specific implementation details.",
                " Be precise about the algorithm complexity.",
                " Specify exact function parameters and return types."
            ],
            "general": [
                " Be specific.",
                " Provide detailed information.",
                " Include precise details."
            ]
        }
        
        markers = specificity_markers.get(task_type, specificity_markers["general"])
        marker = random.choice(markers)
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{marker}"
        else:
            return f"{prompt}.{marker}"
    
    def _add_constraints(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Add constraints to guide the response format."""
        if "includes_constraints" in self._extract_prompt_patterns(prompt):
            return prompt
            
        # Constraints by task type
        constraints = {
            "explanation": [
                " Limit your explanation to 3-5 key points.",
                " Keep your answer concise, under 150 words.",
                " Focus only on the most important aspects."
            ],
            "creative": [
                " Keep this under 200 words.",
                " Include exactly three main characters.",
                " Limit the setting to a single location."
            ],
            "summarization": [
                " Summarize in 3-5 bullet points.",
                " Keep your summary under 100 words.",
                " Focus only on the main arguments."
            ],
            "general": [
                " Keep your answer brief.",
                " Focus on the key points only.",
                " Be concise and to the point."
            ]
        }
        
        constraints_list = constraints.get(task_type, constraints["general"])
        constraint = random.choice(constraints_list)
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{constraint}"
        else:
            return f"{prompt}.{constraint}"
    
    def _add_perspective_request(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Add a request for a specific perspective."""
        if "includes_perspective_request" in self._extract_prompt_patterns(prompt):
            return prompt
            
        if task_type not in ["opinion", "analysis", "comparison", "creative", "general"]:
            return prompt
            
        # Perspective requests by task type
        perspective_requests = {
            "opinion": [
                " Consider different perspectives.",
                " Analyze from multiple viewpoints.",
                " Include contrasting opinions."
            ],
            "analysis": [
                " Consider historical and modern perspectives.",
                " Analyze from technical and practical standpoints.",
                " Include both theoretical and applied perspectives."
            ],
            "comparison": [
                " Compare from multiple perspectives.",
                " Consider advantages and disadvantages from different viewpoints.",
                " Contrast the perspectives of different stakeholders."
            ],
            "creative": [
                " Show this from the perspective of different characters.",
                " Include multiple viewpoints in your narrative.",
                " Explore this from unusual perspectives."
            ],
            "general": [
                " Consider different perspectives.",
                " Include multiple viewpoints.",
                " Show both sides of the issue."
            ]
        }
        
        requests = perspective_requests.get(task_type, perspective_requests["general"])
        request = random.choice(requests)
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{request}"
        else:
            return f"{prompt}.{request}"
    
    def _simplify_structure(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Simplify the structure of complex prompts."""
        # Only apply if prompt is complex
        if "simple_sentence_structure" in self._extract_prompt_patterns(prompt):
            return prompt
            
        # If prompt is long or has complex structure
        if len(prompt) > 100 or ";" in prompt or prompt.count(",") > 3:
            # Extract core request
            sentences = [s.strip() for s in prompt.split(".") if s.strip()]
            
            if len(sentences) > 1:
                main_sentence = sentences[0]
                return f"{main_sentence}. Keep it simple and direct."
            else:
                # Try to simplify by removing modifiers
                words = prompt.split()
                if len(words) > 15:
                    simplified = " ".join(words[:10])
                    return f"{simplified}. Keep it simple."
        
        return prompt  # No simplification needed
    
    def _adjust_length(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Adjust prompt length if too short or too long."""
        if "moderate_length" in self._extract_prompt_patterns(prompt):
            return prompt
            
        if len(prompt) < 30:
            # Too short, add context
            elaborations = {
                "explanation": " Provide a thorough explanation with key concepts defined.",
                "factual": " Include important facts and relevant context.",
                "instruction": " Make sure to include all necessary steps and details.",
                "coding": " Include proper syntax, error handling, and comments.",
                "general": " Please provide a complete answer with sufficient detail."
            }
            
            elaboration = elaborations.get(task_type, elaborations["general"])
            return f"{prompt}{elaboration}"
            
        elif len(prompt) > 250:
            # Too long, truncate with focus directive
            words = prompt.split()
            truncated = " ".join(words[:30])
            
            focus_directives = {
                "explanation": " Focus on explaining the core concept clearly.",
                "factual": " Focus on the most important facts.",
                "instruction": " Focus on the essential steps.",
                "coding": " Focus on the key algorithm or function.",
                "general": " Focus on the main points only."
            }
            
            directive = focus_directives.get(task_type, focus_directives["general"])
            return f"{truncated}...{directive}"
            
        return prompt  # Length is fine
    
    def _formalize_language(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Make language more formal if appropriate for the task."""
        if "formal_language" in self._extract_prompt_patterns(prompt):
            return prompt
            
        # Only formalize for certain task types
        if task_type not in ["factual", "analysis", "explanation", "general"]:
            return prompt
            
        # Replace informal phrases with formal ones
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
            "bad": "detrimental",
            "stuff": "materials",
            "things": "elements"
        }
        
        result = prompt
        for informal, formal in informal_to_formal.items():
            if f" {informal} " in f" {result} ":  # Match whole words only
                result = result.replace(f" {informal} ", f" {formal} ")
                
        # If no replacements were made, add a formal request
        if result == prompt:
            formal_requests = {
                "factual": " Please provide a scholarly response.",
                "analysis": " I would appreciate a rigorous analysis.",
                "explanation": " Please respond with a formal explanation.",
                "general": " I request a formal response."
            }
            
            request = formal_requests.get(task_type, formal_requests["general"])
            
            prompt = prompt.rstrip()
            if prompt.endswith((".", "!", "?")):
                return f"{prompt}{request}"
            else:
                return f"{prompt}.{request}"
            
        return result
    
    def _address_audience(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Add explicit audience targeting to the prompt."""
        if "addresses_audience" in self._extract_prompt_patterns(prompt):
            return prompt
            
        # Audience specifications by task type
        audience_specs = {
            "explanation": [
                " Explain this to a beginner.",
                " Target your explanation for high school students.",
                " Explain as if to someone with no background knowledge."
            ],
            "coding": [
                " Write this for junior developers.",
                " Explain the code for someone new to programming.",
                " Target experienced programmers in your explanation."
            ],
            "factual": [
                " Present this for an educated general audience.",
                " Write this for readers of a science magazine.",
                " Target this for undergraduate students."
            ],
            "general": [
                " Target your response for a general audience.",
                " Prepare your answer for non-experts.",
                " Frame your response for interested beginners."
            ]
        }
        
        specs = audience_specs.get(task_type, audience_specs["general"])
        spec = random.choice(specs)
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{spec}"
        else:
            return f"{prompt}.{spec}"
    
    def _explain_purpose(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Add purpose explanation to the prompt."""
        if "explains_purpose" in self._extract_prompt_patterns(prompt):
            return prompt
            
        # Purpose explanations by task type
        purpose_explanations = {
            "explanation": [
                " I'm asking this to understand the fundamental concepts.",
                " My goal is to grasp the basic principles.",
                " I need this explained to build my knowledge foundation."
            ],
            "comparison": [
                " I need this comparison to make an informed decision.",
                " I'm trying to understand the key differences.",
                " My aim is to evaluate these alternatives."
            ],
            "instruction": [
                " I need these instructions to complete a project.",
                " My goal is to learn this process for practical application.",
                " I want to master this technique."
            ],
            "general": [
                " I'm asking because I need to understand this better.",
                " My goal is to learn about this topic.",
                " I'm trying to expand my knowledge in this area."
            ]
        }
        
        explanations = purpose_explanations.get(task_type, purpose_explanations["general"])
        explanation = random.choice(explanations)
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{explanation}"
        else:
            return f"{prompt}.{explanation}"
    
    def _set_format_expectations(self, prompt: str, analysis: Dict, task_type: str) -> str:
        """Set expectations for response format."""
        if "sets_format_expectations" in self._extract_prompt_patterns(prompt):
            return prompt
            
        # Format expectations by task type
        format_expectations = {
            "explanation": [
                " Structure your answer with headings and bullet points.",
                " Start with a simple overview, then add details.",
                " Use analogies to simplify complex concepts."
            ],
            "instruction": [
                " Present this as a numbered step-by-step guide.",
                " Format with clear sections for each major step.",
                " Include a checklist at the end."
            ],
            "comparison": [
                " Organize this as a side-by-side comparison.",
                " Use a pros and cons format.",
                " Structure as categories with comparisons in each."
            ],
            "coding": [
                " Include commented code blocks.",
                " Structure this as pseudocode first, then implementation.",
                " Include usage examples after the code."
            ],
            "general": [
                " Organize your response clearly with paragraphs or bullet points.",
                " Start with a brief summary, then expand with details.",
                " Format your answer for easy readability."
            ]
        }
        
        expectations = format_expectations.get(task_type, format_expectations["general"])
        expectation = random.choice(expectations)
        
        prompt = prompt.rstrip()
        if prompt.endswith((".", "!", "?")):
            return f"{prompt}{expectation}"
        else:
            return f"{prompt}.{expectation}"