"""
Metrics utilities for PromptSage
"""

import torch
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: Optional[str] = None,
    stride: int = 512
) -> float:
    """
    Calculate perplexity of text using a language model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: The text to calculate perplexity for
        device: Device to run on
        stride: Stride length for processing long texts
        
    Returns:
        Perplexity score (lower is better)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tokenize text
    encodings = tokenizer(text, return_tensors="pt")
    
    # Move to device
    input_ids = encodings.input_ids.to(device)
    
    # For short text, calculate directly
    if input_ids.size(1) <= stride:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            return torch.exp(outputs.loss).item()
    
    # For longer text, use strided approach
    nlls = []
    max_length = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 1024
    
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        
        target_len = end_loc - i  # May be different from stride at the end
        
        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        
        # Set up target labels: -100 ignores loss
        target_chunk = input_chunk.clone()
        target_chunk[:, :-target_len] = -100
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * target_len
        
        nlls.append(neg_log_likelihood)
    
    # Average NLLs and convert to perplexity
    return torch.exp(torch.stack(nlls).sum() / end_loc).item()

def calculate_response_metrics(
    prompt: str,
    response: str,
    reference: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate various metrics for a response.
    
    Args:
        prompt: Original prompt
        response: Model response
        reference: Optional reference response for comparison
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic length metrics
    metrics["response_length"] = len(response.split())
    metrics["prompt_response_ratio"] = len(response.split()) / max(1, len(prompt.split()))
    
    # Lexical diversity (unique tokens / total tokens)
    response_tokens = response.lower().split()
    if response_tokens:
        metrics["lexical_diversity"] = len(set(response_tokens)) / len(response_tokens)
    else:
        metrics["lexical_diversity"] = 0
    
    # If reference is provided, calculate reference-based metrics
    if reference:
        # Basic word overlap (simplified ROUGE-1)
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        
        if reference_words:
            # Precision, recall, F1
            overlap = len(response_words.intersection(reference_words))
            
            # Precision: proportion of response words that appear in reference
            metrics["precision"] = overlap / max(1, len(response_words))
            
            # Recall: proportion of reference words that appear in response
            metrics["recall"] = overlap / len(reference_words)
            
            # F1 score: harmonic mean of precision and recall
            if metrics["precision"] + metrics["recall"] > 0:
                metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
            else:
                metrics["f1"] = 0
                
    return metrics

def calculate_response_similarity(
    response1: str,
    response2: str
) -> float:
    """
    Calculate similarity between two responses.
    
    Args:
        response1: First response
        response2: Second response
        
    Returns:
        Similarity score (0-1)
    """
    # Simple Jaccard similarity
    words1 = set(response1.lower().split())
    words2 = set(response2.lower().split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0
        
    return intersection / union