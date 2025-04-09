"""
Prompt evaluator module for PromptSage
"""

import torch
from typing import Dict, List, Optional, Union
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

class PromptEvaluator:
    """Evaluates prompt quality and response metrics."""
    
    def __init__(
        self, 
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
        device: str = None,
        use_cached_models: bool = True,
    ):
        """
        Initialize the PromptEvaluator.
        
        Args:
            model: Pre-loaded language model
            tokenizer: Pre-loaded tokenizer
            device: Device to run evaluations on
            use_cached_models: Whether to use cached models for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load evaluation models for different metrics
        if use_cached_models:
    # Load smaller models for specific evaluation tasks
         try:
            import transformers
            # For coherence evaluation
            try:
                self.coherence_model = AutoModelForSequenceClassification.from_pretrained(
                "prithivida/coherence_model"
                ).to(self.device)
                self.coherence_tokenizer = AutoTokenizer.from_pretrained("prithivida/coherence_model")
            except Exception as e:
                print(f"Warning: Could not load coherence model: {e}")
                self.coherence_model = None
                self.coherence_tokenizer = None
            
            # For factuality evaluation
            try:
                self.factuality_model = AutoModelForSequenceClassification.from_pretrained(
                    "stanford-crfm/factuality-classifier"
                ).to(self.device)
                self.factuality_tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/factuality-classifier")
            except Exception as e:
                print(f"Warning: Could not load factuality model: {e}")
                self.factuality_model = None
                self.factuality_tokenizer = None
            
         except Exception as e:
            print(f"Warning: Could not load evaluation models: {e}")
            self.coherence_model = None
            self.factuality_model = None
    
    def evaluate(
        self, 
        prompt: str, 
        response: str,
        task_type: str = "general",
        reference: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a prompt and its response.
        
        Args:
            prompt: The prompt to evaluate
            response: The model's response to the prompt
            task_type: Type of task the prompt represents
            reference: Optional reference response for comparison
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Calculate perplexity (lower is better)
        metrics["perplexity"] = self._calculate_perplexity(prompt, response)
        
        # Calculate response metrics
        metrics["response_length"] = len(response.split())
        metrics["prompt_response_ratio"] = len(response.split()) / max(1, len(prompt.split()))
        
        # Calculate coherence score
        if self.coherence_model is not None:
            metrics["coherence_score"] = self._calculate_coherence(prompt, response)
        
        # Calculate factuality score for relevant task types
        if self.factuality_model is not None and task_type in ["explanation", "factual", "definition"]:
            metrics["factuality_score"] = self._calculate_factuality(response)
        
        # Calculate relevance score
        metrics["relevance_score"] = self._calculate_relevance(prompt, response)
        
        # Calculate specificity score
        metrics["specificity_score"] = self._calculate_specificity(response)
        
        # If we have a reference, calculate reference-based metrics
        if reference is not None:
            ref_metrics = self._calculate_reference_metrics(response, reference)
            metrics.update(ref_metrics)
        
        # Task-specific metrics
        if task_type == "creative":
            metrics["creativity_score"] = self._calculate_creativity(response)
            
        elif task_type == "coding":
            metrics["code_quality_score"] = self._calculate_code_quality(response)
            
        elif task_type == "instruction":
            metrics["clarity_score"] = self._calculate_instruction_clarity(response)
        
        return metrics
    
    def _calculate_perplexity(self, prompt: str, response: str) -> float:
        """Calculate perplexity of the response given the prompt."""
        if self.model is None or self.tokenizer is None:
            # Return placeholder if model not available
            return 0.0
            
        # Combine prompt and response
        full_text = prompt + " " + response
        
        with torch.no_grad():
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            
            # Get prompt length to ignore in loss calculation
            prompt_length = len(self.tokenizer.encode(prompt))
            
            # Calculate loss on response only
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_length] = -100  # Ignore prompt tokens
            
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            # Convert loss to perplexity
            perplexity = torch.exp(loss).item()
            
        return perplexity
    
    def _calculate_coherence(self, prompt: str, response: str) -> float:
        """Calculate coherence score between prompt and response."""
        if self.coherence_model is None:
            return 0.7  # Default placeholder
            
        # Combine prompt and response
        text_pair = [prompt, response]
        
        with torch.no_grad():
            inputs = self.coherence_tokenizer(
                text_pair[0], text_pair[1], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            outputs = self.coherence_model(**inputs)
            coherence_score = torch.softmax(outputs.logits, dim=1)[0, 1].item()  # Positive class probability
            
        return coherence_score
    
    def _calculate_factuality(self, response: str) -> float:
        """Calculate factuality score of the response."""
        if self.factuality_model is None:
            return 0.6  # Default placeholder
            
        with torch.no_grad():
            inputs = self.factuality_tokenizer(
                response, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            outputs = self.factuality_model(**inputs)
            factuality_score = torch.softmax(outputs.logits, dim=1)[0, 1].item()  # Factual class probability
            
        return factuality_score
    
    def _calculate_relevance(self, prompt: str, response: str) -> float:
        """
        Calculate relevance score between prompt and response.
        Uses a simple keyword matching approach as fallback if no embedding model available.
        """
        # Extract keywords from prompt
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Filter out common stop words (simplified approach)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        prompt_keywords = prompt_words - stop_words
        response_keywords = response_words - stop_words
        
        # Calculate overlap
        if not prompt_keywords:
            return 0.5  # Default if no keywords found
            
        overlap = len(prompt_keywords.intersection(response_keywords)) / len(prompt_keywords)
        
        # Scale to a reasonable range
        relevance_score = min(1.0, 0.4 + (overlap * 0.6))
        
        return relevance_score
    
    def _calculate_specificity(self, response: str) -> float:
        """Calculate specificity of the response based on linguistic features."""
        # Count specific features like numbers, proper nouns, specificity indicators
        has_numbers = any(char.isdigit() for char in response)
        has_proper_nouns = any(word[0].isupper() and word[1:].islower() for word in response.split() if len(word) > 1)
        
        # Count specific detail words
        specificity_indicators = [
            "specifically", "in particular", "exactly", "precisely", "detailed", 
            "uniquely", "distinct", "explicit", "definite", "concrete"
        ]
        specificity_count = sum(1 for indicator in specificity_indicators if indicator in response.lower())
        
        # Base score calculation
        score = 0.5  # Default middle score
        
        if has_numbers:
            score += 0.1
        if has_proper_nouns:
            score += 0.1
        
        # Add proportionally for specificity words
        score += min(0.2, specificity_count * 0.05)
        
        # Word variety as a proxy for specificity
        unique_words = len(set(response.lower().split())) / max(1, len(response.split()))
        score += min(0.1, unique_words * 0.2)
        
        return min(1.0, score)
    
    def _calculate_reference_metrics(self, response: str, reference: str) -> Dict[str, float]:
        """
        Calculate reference-based metrics (e.g., ROUGE, BLEU).
        Simplified implementations for compact code.
        """
        metrics = {}
        
        # Basic word overlap (simplified ROUGE-1)
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        
        # Calculate precision, recall, F1
        if response_words and reference_words:
            overlap = len(response_words.intersection(reference_words))
            precision = overlap / max(1, len(response_words))
            recall = overlap / max(1, len(reference_words))
            f1 = 2 * precision * recall / max(0.01, precision + recall)
            
            metrics["word_overlap_precision"] = precision
            metrics["word_overlap_recall"] = recall
            metrics["word_overlap_f1"] = f1
        
        return metrics
    
    def _calculate_creativity(self, response: str) -> float:
        """Calculate creativity score for creative tasks."""
        # This is a naive implementation
        # Real creativity scoring would use more sophisticated methods
        
        # 1. Check for diverse vocabulary
        unique_words_ratio = len(set(response.lower().split())) / max(1, len(response.split()))
        
        # 2. Check for figurative language
        figurative_indicators = [
            "like", "as", "metaphor", "imagine", "similar", 
            "analogy", "comparison", "symbolize", "represent"
        ]
        figurative_count = sum(1 for indicator in figurative_indicators if indicator in response.lower())
        
        # 3. Sentence length variety
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            length_variety = np.std(sent_lengths) / max(1, np.mean(sent_lengths))
        else:
            length_variety = 0
            
        # Combine metrics
        creativity_score = (
            (unique_words_ratio * 0.4) + 
            (min(0.4, figurative_count * 0.1)) + 
            (min(0.2, length_variety * 0.5))
        )
        
        return min(1.0, creativity_score)
    
    def _calculate_code_quality(self, response: str) -> float:
        """Calculate code quality score for coding tasks."""
        # Simplified code quality metrics
        
        # Check if it looks like code at all
        code_indicators = ["def ", "class ", "function", "import ", "return ", "if ", "for ", "while "]
        is_code = any(indicator in response for indicator in code_indicators)
        
        if not is_code:
            return 0.2  # Not code
            
        # Check for code structure
        has_indentation = "    " in response or "\t" in response
        has_comments = "#" in response or "/*" in response or "//" in response
        has_function_def = "def " in response or "function" in response
        
        # Calculate basic score
        score = 0.5  # Base score
        
        if has_indentation:
            score += 0.2
        if has_comments:
            score += 0.15
        if has_function_def:
            score += 0.15
            
        return min(1.0, score)
    
    def _calculate_instruction_clarity(self, response: str) -> float:
        """Calculate clarity score for instruction tasks."""
        # Check for step-by-step structure
        has_numbered_steps = bool(re.search(r"\d+\.\s", response))
        has_bullet_points = "*" in response or "-" in response
        
        # Check for clarity indicators
        clarity_indicators = ["first", "then", "next", "finally", "lastly", "follow", "steps"]
        clarity_count = sum(1 for indicator in clarity_indicators if indicator in response.lower())
        
        # Base score
        score = 0.5
        
        if has_numbered_steps:
            score += 0.25
        elif has_bullet_points:
            score += 0.15
            
        score += min(0.25, clarity_count * 0.05)
        
        return min(1.0, score)