"""Adaptive `balance` backoff for AutoNFS.

Rationale (see the HPO study, `AutoNFS_HPO_report.md` / addendum): `balance=1.0`
maximizes predictive power *per selected feature* (the most aggressive,
smallest-subset selection), but on several benchmark datasets it collapses the
mask entirely (0 features selected). This module implements a per-dataset
backoff: start at `balance=1.0` and only relax it, one log-grid step at a
time, until the mask no longer collapses and downstream score does not fall
too far below an all-features reference.

Used internally by `AutoNFS(balance="auto")` (see `sklearn_interface.py`); can
also be called directly for diagnostics (e.g. to plot the backoff trajectory).
"""
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

DEFAULT_GRID = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)


@dataclass
class BalanceGridPoint:
    balance: float
    median_n_selected: float
    median_score: float
    n_selected_all: list
    scores_all: list
    meets_criteria: bool


@dataclass
class AdaptiveBalanceResult:
    chosen_balance: float
    chosen_reason: str  # "criteria_met" | "fallback_best" | "fallback_last"
    trajectory: list = field(default_factory=list)


def _baseline_score(X_train, y_train, X_val, y_val, mode):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import balanced_accuracy_score, r2_score

    if mode == "classification":
        clf = RandomForestClassifier(random_state=0, n_estimators=200, n_jobs=-1)
        clf.fit(X_train, y_train)
        return balanced_accuracy_score(y_val, clf.predict(X_val))
    else:
        clf = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
        clf.fit(X_train, y_train)
        return r2_score(y_val, clf.predict(X_val))


def _downstream_score(X_train, y_train, X_val, y_val, support, mode):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import balanced_accuracy_score, r2_score

    if support.sum() == 0:
        return float("nan")
    Xtr_sel, Xval_sel = X_train[:, support], X_val[:, support]
    if mode == "classification":
        clf = RandomForestClassifier(random_state=0, n_estimators=200, n_jobs=-1)
        clf.fit(Xtr_sel, y_train)
        return balanced_accuracy_score(y_val, clf.predict(Xval_sel))
    else:
        clf = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
        clf.fit(Xtr_sel, y_train)
        return r2_score(y_val, clf.predict(Xval_sel))


def adaptive_balance_search(
    X_train,
    y_train,
    X_val,
    y_val,
    mode: Literal["classification", "regression"] = "classification",
    grid=DEFAULT_GRID,
    n_seeds: int = 5,
    threshold_frac: float = 0.9,
    batch_size: int = 32,
    epochs: int = 150,
    temperature_decay: float = 0.997,
    device: str = "cpu",
    baseline_score: Optional[float] = None,
    verbose: bool = False,
) -> AdaptiveBalanceResult:
    """Find the largest `balance` (most feature-selection pressure) that does not
    collapse the mask (median #selected == 0 over `n_seeds` seeds) and does not
    lose more than `1 - threshold_frac` of an all-features baseline's downstream
    score (median over seeds), on a held-out (`X_val`, `y_val`) split.

    Walks `grid` from the most aggressive value down, stopping at the first
    value that clears both criteria. If none does, falls back to the grid
    point with the best median score among non-collapsed points, or the
    smallest grid value if every point collapsed.
    """
    import torch
    from .ensemble import train_gumbel_ensemble

    if baseline_score is None:
        baseline_score = _baseline_score(X_train, y_train, X_val, y_val, mode)

    # Prepare tensors once, exactly as AutoNFS.fit(scale=True) would per run.
    X_t = torch.as_tensor(X_train)
    y_t = torch.as_tensor(y_train)
    if mode == "classification":
        y_t = torch.nn.functional.one_hot(y_t.to(int))
    else:
        y_t = y_t.view(-1, 1).to(torch.float32)
    X_t = (X_t - X_t.mean(0)) / (X_t.std(0) + 1e-6)

    trajectory = []
    chosen, reason = None, None

    for balance in grid:
        n_selected_all, scores_all = [], []
        # One vectorized training of n_seeds stacked networks; equivalent to
        # n_seeds sequential AutoNFS fits (see ensemble.py) but ~n_seeds x faster.
        votes = train_gumbel_ensemble(
            X_t, y_t, n_members=n_seeds,
            batch_size=batch_size, epochs=epochs, fs_balance=balance,
            temperature_decay=temperature_decay, device=device, mode=mode,
        )
        # The scoring forest is deterministic (random_state=0), so seeds that
        # picked the same support get the same score -- compute it once.
        score_cache = {}
        for seed in range(n_seeds):
            support = votes[seed] > 0
            n_selected_all.append(int(support.sum()))
            key = support.tobytes()
            if key not in score_cache:
                score_cache[key] = _downstream_score(X_train, y_train, X_val, y_val, support, mode)
            scores_all.append(score_cache[key])

        median_n = float(np.median(n_selected_all))
        scores_for_median = [0.0 if np.isnan(s) else s for s in scores_all]
        median_score = float(np.median(scores_for_median))
        meets = (median_n > 0) and (median_score >= threshold_frac * baseline_score)

        point = BalanceGridPoint(
            balance=balance, median_n_selected=median_n, median_score=median_score,
            n_selected_all=n_selected_all, scores_all=scores_all, meets_criteria=meets,
        )
        trajectory.append(point)

        if verbose:
            print(f"balance={balance:<8g} median_n={median_n:.1f} "
                  f"median_score={median_score:.4f} meets={meets}")

        if meets:
            chosen, reason = balance, "criteria_met"
            break

    if chosen is None:
        candidates = [p for p in trajectory if p.median_n_selected > 0]
        if candidates:
            best = max(candidates, key=lambda p: p.median_score)
            chosen, reason = best.balance, "fallback_best"
        else:
            chosen, reason = trajectory[-1].balance, "fallback_last"

    return AdaptiveBalanceResult(chosen_balance=chosen, chosen_reason=reason, trajectory=trajectory)
