import numpy as np
from typing import List

class StabilityMonitor:
    """
    Implements a statistical monitor to detect opinion stability 
    and trigger an early halt to the debate.
    """
    
    def __init__(self, window_size: int = 3, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.scores_history = []

    def check_stability(self, current_scores: List[float]) -> bool:
        """
        Check if the distribution of agent stances has stabilized.
        In this implementation, we use the variance of consensus scores
        over a sliding window as a proxy for stability.
        """
        if not current_scores:
            return False
            
        avg_score = np.mean(current_scores)
        self.scores_history.append(avg_score)
        
        if len(self.scores_history) < self.window_size:
            return False
            
        # Calculate variance over the window
        window = self.scores_history[-self.window_size:]
        variance = np.var(window)
        
        # If variance is below threshold, the debate has stabilized
        return variance < self.threshold

def calculate_consensus_delta(scores: List[float]) -> float:
    """Helper to find the max spread between agents."""
    if not scores:
        return 0.0
    return max(scores) - min(scores)
