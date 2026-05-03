"""Evaluation metrics for multi-agent debate system."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
import re

# Load embedding model (lazy load)
_embedding_model = None

def get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def calculate_cosine_dissimilarity(text1: str, text2: str) -> float:
    """Calculate cosine dissimilarity (1 - cosine similarity) between two texts."""
    model = get_embedding_model()
    
    # Generate embeddings
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    
    # Return dissimilarity (1 - similarity)
    return 1.0 - similarity

def calculate_information_gain(responses: List[str]) -> List[float]:
    """
    Calculate information gain between consecutive responses.
    Returns a list of dissimilarity scores (higher = more new information).
    """
    if len(responses) < 2:
        return []
    
    dissimilarities = []
    for i in range(len(responses) - 1):
        dissim = calculate_cosine_dissimilarity(responses[i], responses[i + 1])
        dissimilarities.append(dissim)
    
    return dissimilarities

def detect_repetitive_loop(responses: List[str], threshold: float = 0.2) -> bool:
    """
    Detect if responses are falling into a repetitive loop.
    Returns True if average dissimilarity is below threshold.
    """
    dissimilarities = calculate_information_gain(responses)
    
    if not dissimilarities:
        return False
    
    avg_dissimilarity = np.mean(dissimilarities)
    return avg_dissimilarity < threshold

def calculate_turn_faithfulness(argument: str, search_results: str, threshold: float = 0.5) -> float:
    """
    Calculate the percentage of argument sentences that are grounded in search results.
    """
    if not search_results or not argument:
        return 0.0
        
    model = get_embedding_model()
    
    # Simple sentence splitting
    arg_sentences = [s.strip() for s in re.split(r'[.!?]+', argument) if len(s.strip()) > 10]
    search_sentences = [s.strip() for s in re.split(r'[.!?]+', search_results) if len(s.strip()) > 10]
    
    if not arg_sentences or not search_sentences:
        return 0.0
        
    arg_embs = model.encode(arg_sentences)
    search_embs = model.encode(search_sentences)
    
    # Calculate similarity matrix
    sim_matrix = cosine_similarity(arg_embs, search_embs)
    
    # For each argument sentence, find max similarity with any search sentence
    max_sims = np.max(sim_matrix, axis=1)
    
    # Count how many sentences are above threshold
    faithful_count = np.sum(max_sims >= threshold)
    
    return float(faithful_count / len(arg_sentences))

class DebateMetrics:
    """Track and calculate debate evaluation metrics."""
    
    def __init__(self):
        self.proposer_responses: List[str] = []
        self.critic_responses: List[str] = []
        self.information_gains: List[float] = []
        self.position_swap_scores: List[Dict[str, Any]] = []
        self.turn_faithfulness: List[float] = []
        self.format_adherence: Dict[str, int] = {"valid": 0, "total": 0}
        self.search_efficiency: Dict[str, int] = {"total_searches": 0, "redundant_searches": 0, "empty_searches": 0}
    
    def add_proposer_response(self, response: str):
        """Add a proposer response and calculate information gain."""
        if self.proposer_responses:
            dissim = calculate_cosine_dissimilarity(
                self.proposer_responses[-1], 
                response
            )
            self.information_gains.append(dissim)
        self.proposer_responses.append(response)
    
    def add_critic_response(self, response: str):
        """Add a critic response and calculate information gain."""
        if self.critic_responses:
            dissim = calculate_cosine_dissimilarity(
                self.critic_responses[-1], 
                response
            )
            self.information_gains.append(dissim)
        self.critic_responses.append(response)
    
    def get_average_information_gain(self) -> float:
        """Get average information gain across all responses."""
        if not self.information_gains:
            return 0.0
        return np.mean(self.information_gains)
    
    def is_repetitive_loop(self, threshold: float = 0.2) -> bool:
        """Check if the debate is in a repetitive loop."""
        all_responses = []
        for p, c in zip(self.proposer_responses, self.critic_responses):
            all_responses.append(p)
            all_responses.append(c)
        return detect_repetitive_loop(all_responses, threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "average_information_gain": float(self.get_average_information_gain()),
            "is_repetitive_loop": bool(self.is_repetitive_loop()),
            "information_gains": [float(x) for x in self.information_gains],
            "num_proposer_responses": len(self.proposer_responses),
            "num_critic_responses": len(self.critic_responses),
            "position_swap_scores": self.position_swap_scores,
            "turn_faithfulness": [float(x) for x in self.turn_faithfulness],
            "format_adherence": self.format_adherence,
            "format_adherence_percent": (self.format_adherence["valid"] / self.format_adherence["total"] * 100) if self.format_adherence["total"] > 0 else 0,
            "search_efficiency": self.search_efficiency
        }
