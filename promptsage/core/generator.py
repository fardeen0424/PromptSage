"""
Prompt generator module for PromptSage
"""

import torch
from typing import List, Dict, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

class PromptGenerator:
    """Generates responses and optimized prompts."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
        device: str = None,
    ):
        """
        Initialize the PromptGenerator.
        
        Args:
            model: Pre-loaded language model
            tokenizer: Pre-loaded tokenizer
            device: Device to run the generator on
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device    
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> Union[str, List[str]]:
        """
        Generate a response from a prompt.
        
        Args:
            prompt: The prompt to generate from
            max_length: Maximum length of generated text
            temperature: Generation temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            do_sample: Whether to use sampling (vs greedy decoding)
            
        Returns:
            Generated text (str) or list of texts if num_return_sequences > 1
        """
        if self.model is None or self.tokenizer is None:
            # Return placeholder if model not available
            if num_return_sequences > 1:
                return ["[Generated response placeholder]"] * num_return_sequences
            return "[Generated response placeholder]"
        
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Generate
        with torch.no_grad():
            output_sequences = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in output_sequences
        ]
        
        # Remove the prompt from the beginning of each generated text
        prompt_length = len(prompt)
        generated_texts = [
            text[prompt_length:].strip() if text.startswith(prompt) else text
            for text in generated_texts
        ]
        
        # Return single string if only one sequence requested
        if num_return_sequences == 1:
            return generated_texts[0]
        
        return generated_texts
    
    def generate_optimized_prompt(
        self,
        original_prompt: str,
        task_type: str = "general",
        target_audience: str = "general",
        optimization_goals: List[str] = None,
    ) -> str:
        """
        Generate an optimized version of a prompt directly.
        
        Args:
            original_prompt: The original prompt to optimize
            task_type: Type of task
            target_audience: Target audience
            optimization_goals: List of optimization goals
            
        Returns:
            Optimized prompt
        """
        if optimization_goals is None:
            optimization_goals = ["clarity", "specificity"]
        
        # Create a meta-prompt to optimize the prompt
        meta_prompt = self._create_meta_prompt(
            original_prompt, task_type, target_audience, optimization_goals
        )
        
        # Generate the optimized prompt
        optimized_prompt = self.generate(
            meta_prompt, max_length=len(original_prompt) * 2, temperature=0.7
        )
        
        # Extract the actual optimized prompt from the response
        # (assuming the response format follows our meta-prompt instructions)
        optimized_prompt = self._extract_optimized_prompt(optimized_prompt)
        
        return optimized_prompt
    
    def _create_meta_prompt(
        self,
        original_prompt: str,
        task_type: str,
        target_audience: str,
        optimization_goals: List[str],
    ) -> str:
        """Create a meta-prompt to optimize another prompt."""
        goals_text = ", ".join(optimization_goals)
        
        meta_prompt = f"""
        Your task is to improve the following prompt for a {task_type} task aimed at {target_audience}.
        
        Original prompt: "{original_prompt}"
        
        Please optimize this prompt for: {goals_text}.
        
        The optimized prompt should maintain the original intent but be more effective.
        
        Optimized prompt:
        """
        
        return meta_prompt.strip()
    
    def _extract_optimized_prompt(self, generated_text: str) -> str:
        """Extract the optimized prompt from the generated response."""
        # Try to find an optimized prompt in quotes
        import re
        
        # Look for text in quotes
        quote_match = re.search(r'"([^"]+)"', generated_text)
        if quote_match:
            return quote_match.group(1)
        
        # If no quotes, look for text after common prefixes
        prefixes = [
            "Optimized prompt:",
            "Improved prompt:",
            "Here's an optimized version:",
            "Here is the optimized prompt:"
        ]
        
        for prefix in prefixes:
            if prefix in generated_text:
                return generated_text.split(prefix)[1].strip()
        
        # If no structure detected, just return the generated text
        return generated_text.strip()