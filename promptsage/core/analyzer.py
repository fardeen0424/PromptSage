"""
Prompt analyzer module for PromptSage
"""

import re
from typing import Dict, List, Optional, Tuple
# Add this near the top of analyzer.py
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

class PromptAnalyzer:
    """Analyzes prompts for various characteristics and patterns."""
    
    def __init__(self):
        """Initialize the PromptAnalyzer with required models and resources."""
        self.ambiguous_phrases = [
            "maybe", "perhaps", "might", "kind of", "sort of", 
            "approximately", "around", "about", "roughly", "generally"
        ]
        self.vague_words = [
            "thing", "stuff", "something", "good", "bad", "nice", 
            "interesting", "various", "several", "many", "few", "some"
        ]
        
    def analyze(self, prompt: str) -> Dict:
        """
        Analyze a prompt and return various characteristics.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Basic text statistics
        word_count = len(prompt.split())
        char_count = len(prompt)
        sentence_count = len(nltk.sent_tokenize(prompt))
        avg_word_length = sum(len(word) for word in prompt.split()) / max(1, word_count)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Lexical analysis
        tokens = nltk.word_tokenize(prompt)
        pos_tags = nltk.pos_tag(tokens)
        
        # Count POS categories
        pos_counts = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        
        # Calculate noun to verb ratio (indicator of complexity)
        noun_count = sum(pos_counts.get(tag, 0) for tag in ["NN", "NNS", "NNP", "NNPS"])
        verb_count = sum(pos_counts.get(tag, 0) for tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])
        noun_verb_ratio = noun_count / max(1, verb_count)
        
        # Detect question types
        is_question = prompt.strip().endswith("?")
        has_wh_question = any(tag in ["WP", "WRB", "WDT", "WP$"] for _, tag in pos_tags)
        
        # Check for prompt patterns
        has_instructions = self._contains_instructions(prompt)
        specificity_score = self._calculate_specificity(prompt)
        clarity_score = self._calculate_clarity(prompt, pos_tags)
        
        # Task inference
        likely_tasks = self._infer_task_type(prompt, pos_tags)
        
        return {
            # Basic statistics
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            
            # Linguistic features
            "pos_distribution": {tag: count/max(1, len(pos_tags)) for tag, count in pos_counts.items()},
            "noun_verb_ratio": noun_verb_ratio,
            
            # Question features
            "is_question": is_question,
            "has_wh_question": has_wh_question,
            
            # Quality metrics
            "has_instructions": has_instructions,
            "specificity_score": specificity_score,
            "clarity_score": clarity_score,
            
            # Task inference
            "likely_tasks": likely_tasks,
        }
    
    def _contains_instructions(self, prompt: str) -> bool:
        """Check if the prompt contains explicit instructions."""
        instruction_patterns = [
            r'\b(?:explain|describe|list|enumerate|outline|summarize|analyze)\b',
            r'\b(?:provide|give|offer|present|show)\b.{0,20}\b(?:example|explanation|reason|detail)',
            r'\b(?:how|what|why|when|where|who)\b.{0,30}\b(?:can|could|would|should|will|shall)',
        ]
        
        return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in instruction_patterns)
    
    def _calculate_specificity(self, prompt: str) -> float:
        """Calculate a specificity score for the prompt."""
        # Check for specific details
        has_numbers = bool(re.search(r'\d+', prompt))
        has_proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', prompt)) > 0
        has_specific_details = len(re.findall(r'\b(?:specifically|in particular|exactly|precisely)\b', prompt)) > 0
        
        # Check for vague language
        vague_word_count = sum(1 for word in self.vague_words if re.search(r'\b' + word + r'\b', prompt, re.IGNORECASE))
        vague_word_penalty = min(0.5, vague_word_count * 0.1)
        
        # Base score from 0.5
        score = 0.5
        
        # Add for specificity indicators
        if has_numbers:
            score += 0.15
        if has_proper_nouns:
            score += 0.15
        if has_specific_details:
            score += 0.1
            
        # Subtract for vagueness
        score = max(0.1, score - vague_word_penalty)
        
        # Normalize to 0-1
        return round(min(1.0, score), 2)
    
    def _calculate_clarity(self, prompt: str, pos_tags: List[Tuple[str, str]]) -> float:
        """Calculate a clarity score for the prompt."""
        # Check for ambiguous language
        ambiguous_phrase_count = 0
        for phrase in self.ambiguous_phrases:
            ambiguous_phrase_count += len(re.findall(r'\b' + re.escape(phrase) + r'\b', prompt, re.IGNORECASE))
        
        # Check sentence structure complexity
        complex_conjunction_count = len(re.findall(r'\b(?:although|though|however|despite|nevertheless)\b', prompt, re.IGNORECASE))
        
        # Check for passive voice
        passive_constructs = 0
        for i in range(len(pos_tags) - 2):
            # Simple passive detection: be verb + past participle
            if pos_tags[i][1] in ["VBZ", "VBP", "VBD", "VBN"] and pos_tags[i][0].lower() in ["is", "are", "was", "were", "be", "been", "being"] and pos_tags[i+1][1] == "VBN":
                passive_constructs += 1
        
        # Base score from the high end
        score = 0.9
        
        # Subtract for clarity issues
        ambiguity_penalty = min(0.3, ambiguous_phrase_count * 0.05)
        complexity_penalty = min(0.2, complex_conjunction_count * 0.05)
        passive_penalty = min(0.2, passive_constructs * 0.1)
        
        score = max(0.1, score - ambiguity_penalty - complexity_penalty - passive_penalty)
        
        # Normalize to 0-1
        return round(min(1.0, score), 2)
    
    def _infer_task_type(self, prompt: str, pos_tags: List[Tuple[str, str]]) -> List[str]:
        """Infer the likely task type from the prompt."""
        task_indicators = {
            "explanation": [r'\bexplain\b', r'\bdescribe\b', r'\bwhat\s+is\b', r'how\s+does\b'],
            "comparison": [r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bvs\b', r'\bsimilarities\b', r'\bdistinguish\b'],
            "opinion": [r'\bopinion\b', r'\bthink\b', r'\bfeel\b', r'\bbelieve\b', r'\byour\s+thoughts\b'],
            "instruction": [r'\bhow\s+to\b', r'\bsteps\b', r'\bguide\b', r'\binstruct\b', r'\bprocess\b'],
            "creative": [r'\bcreate\b', r'\bwrite\b', r'\bimagine\b', r'\bstory\b', r'\bpoem\b', r'\bscene\b'],
            "factual": [r'\bfacts\b', r'\bhistory\b', r'\binformation\b', r'\bdata\b', r'\bwhen\s+did\b'],
            "summarization": [r'\bsummarize\b', r'\bsummary\b', r'\boverview\b', r'\brecap\b'],
            "analysis": [r'\banalyze\b', r'\banalysis\b', r'\bexamine\b', r'\bevaluate\b', r'\binterpret\b'],
            "prediction": [r'\bpredict\b', r'\bforecast\b', r'\bfuture\b', r'\bwill\s+happen\b', r'\bexpect\b'],
            "definition": [r'\bdefine\b', r'\bmeaning\b', r'\bdefinition\b', r'\bwhat\s+does\s+\w+\s+mean\b'],
            "problem-solving": [r'\bsolve\b', r'\bsolution\b', r'\bproblem\b', r'\bresolve\b', r'\bovercome\b'],
            "coding": [r'\bcode\b', r'\bprogram\b', r'\bfunction\b', r'\balgorithm\b', r'\bprogramming\b'],
        }
        
        likely_tasks = []
        for task, patterns in task_indicators.items():
            if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in patterns):
                likely_tasks.append(task)
        
        # If no specific tasks identified, use a default
        if not likely_tasks:
            likely_tasks = ["general"]
        
        return likely_tasks