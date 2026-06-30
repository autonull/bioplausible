from bioplausible.analysis.results import (compute_pareto_frontier,
                                           compute_statistics, get_rankings,
                                           load_trials, load_trials_timeseries)


def format_for_frontend(trials, pareto_ids):
    """
    Format raw trials for the frontend display.
    """
    stats = compute_statistics(trials)
    rankings = get_rankings(trials)

    # Convert list of IDs to set for O(1) lookup
    pareto_set = set(pareto_ids) if pareto_ids else set()

    # Mark Pareto
    for trial in trials:
        trial["is_pareto"] = trial["trial_id"] in pareto_set

    # Best per model
    best_per_model = {}
    from collections import defaultdict

    model_trials = defaultdict(list)
    for trial in trials:
        model_trials[trial["model_name"]].append(trial)

    for model, t_list in model_trials.items():
        best_per_model[model] = max(t_list, key=lambda t: t["accuracy"])

    return {
        "trials": trials,
        "pareto_ids": list(pareto_set),
        "statistics": stats,
        "best_per_model": best_per_model,
        "rankings": rankings,
    }
