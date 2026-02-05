
import pandas as pd
import sqlite3
from typing import Dict, Any, List

class ResearchSynthesizer:
    """
    Synthesizes research insights from experimental results.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def synthesize_full_report(self) -> Dict[str, Any]:
        """Generate all insights."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load Data
            try:
                trials_query = "SELECT * FROM trials WHERE state='completed'"
                trials_df = pd.read_sql(trials_query, conn)
            except Exception:
                trials_df = pd.DataFrame()
                
            try:
                failures_query = "SELECT * FROM failures"
                failures_df = pd.read_sql(failures_query, conn)
            except Exception:
                failures_df = pd.DataFrame()
            
            insights = {
                "cross_algorithm_insights": self._analyze_cross_algo(trials_df),
                "failure_analysis": self._analyze_failures(failures_df),
                "quick_wins": self._find_quick_wins(trials_df, failures_df)
            }
            conn.close()
            return insights
        except Exception as e:
            return {"error": str(e)}

    def _analyze_cross_algo(self, df):
        if df.empty: return "No data available."
        
        try:
            # Group by model family (assuming naming convention or config analysis)
            # Simplified: just group by model_name
            # If accuracy is present
            if "accuracy" in df.columns:
                summary = df.groupby("model_name")["accuracy"].agg(["mean", "max", "count"]).sort_values("max", ascending=False)
                return summary.to_dict()
            return "Trials exist but missing accuracy column."
        except Exception as e:
            return f"Analysis failed: {e}"
        
    def _analyze_failures(self, df):
        if df.empty: return "No failures recorded."
        
        try:
            if "failure_type" in df.columns:
                counts = df["failure_type"].value_counts().to_dict()
                return {"counts": counts}
            return "Failures exist but missing failure_type."
        except Exception as e:
            return f"Failure analysis failed: {e}"
        
    def _find_quick_wins(self, trials, failures):
        suggestions = []
        if not failures.empty and "failure_type" in failures.columns:
             nan_fails = failures[failures["failure_type"].astype(str).str.contains("nan", case=False, na=False)]
             if len(nan_fails) > 0:
                 suggestions.append(f"Suggestion: {len(nan_fails)} trials failed with NaNs. Consider lowering Learning Rates or tightening Gradient Clipping.")
        
        if trials.empty:
            suggestions.append("Suggestion: No successful trials yet. Run more smoke tests.")
            
        return suggestions
