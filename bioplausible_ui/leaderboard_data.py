from bioplausible.analysis.results import (compute_pareto_frontier,
                                           compute_statistics, get_rankings,
                                           load_trials, load_trials_timeseries)

# Actually re-implement format_for_frontend by delegating


def format_for_frontend(trials, pareto_ids):
    stats = compute_statistics(trials)
    rankings = get_rankings(trials)

    # Mark Pareto
    for trial in trials:
        trial["is_pareto"] = trial["trial_id"] in pareto_ids

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
        "pareto_ids": pareto_ids,
        "statistics": stats,
        "best_per_model": best_per_model,
        "rankings": rankings,
    }
