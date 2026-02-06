"""
Curriculum Learning Manager.
Manages the progression of tasks from simple to complex.
"""

class CurriculumManager:
    """
    Defines task tracks and progressions.
    """
    
    # Define tracks
    TRACKS = {
        "vision": ["mnist", "fashion_mnist", "cifar10"],
        "lm": ["char_ngram", "tiny_shakespeare"],
        "rl": ["cartpole", "pendulum", "acrobot"] # Pendulum is arguably harder than cartpole balance
    }
    
    def __init__(self):
        pass
        
    def get_next_task(self, model_family: str, current_task: str, success: bool) -> str:
        """
        Suggest next task based on current outcome.
        """
        # Identify track
        track = None
        for t_name, t_list in self.TRACKS.items():
            if current_task in t_list:
                track = t_list
                break
                
        if not track:
            return None # Unknown track
            
        try:
            curr_idx = track.index(current_task)
        except ValueError:
            return None
            
        if success:
            # Promote
            if curr_idx + 1 < len(track):
                return track[curr_idx + 1]
            else:
                return "completed_track"
        else:
            # Demote or Retry (return None means stay/retry logic handled elsewhere)
            if curr_idx > 0:
                # Optional: Demote if failing hard? 
                # For now, Scientist logic handles retries.
                pass
                
        return None
        
    def get_initial_task(self, model_family: str) -> str:
        """Get starting task for a model family."""
        # Heuristics based on model type
        family = model_family.lower()
        if "transformer" in family or "lm" in family or "language" in family:
            return "char_ngram"
        elif "rl" in family or "control" in family:
            return "cartpole"
        else:
            return "mnist" # Default for general purpose
