"""Regression tests for the ensemble stability-selection default.

See the "Stability selection" section of the README for the cross-dataset
study (37 datasets x 5 seeds) motivating `stability_selection=True` as the
default: it raises mean downstream balanced accuracy vs. the single-run
`balance="auto"` default and eliminates residual mask collapses that the
single-run vote still produced on some datasets even with adaptive balance.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from autonfs import AutoNFS

DEVICE = "cpu"


def test_stability_selection_is_default():
    gfs = AutoNFS()
    assert gfs.stability_selection is True
    assert gfs.n_members == 9
    assert abs(gfs.stability_tau - 1 / 3) < 1e-9


def test_stability_selection_no_collapse_across_seeds():
    """A small, noisy synthetic dataset -- the regime where the single-run
    vote was found to still collapse (k=0) on some seeds even under
    balance="auto". The ensemble stability vote should never collapse.
    """
    X, y = make_classification(
        n_samples=120, n_features=40, n_informative=5, n_redundant=5,
        n_clusters_per_class=2, flip_y=0.15, random_state=0,
    )
    for seed in range(3):
        gfs = AutoNFS(device=DEVICE, random_state=seed, epochs=60)
        gfs.fit(X, y)
        assert gfs.support_.sum() > 0, f"mask collapsed at seed={seed}"


def test_legacy_single_run_still_available():
    """stability_selection=False must reproduce the pre-existing single-run
    behavior (one select_gumbel_features call, self.network populated).
    """
    breast = load_breast_cancer()
    X, y = breast.data, breast.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    gfs = AutoNFS(device=DEVICE, stability_selection=False)
    gfs.fit(X_train, y_train)
    assert gfs.network is not None
    assert gfs.support_.sum() > 0

    clf = RandomForestClassifier(random_state=42)
    clf.fit(gfs.transform(X_train), y_train)
    score = balanced_accuracy_score(y_test, clf.predict(gfs.transform(X_test)))
    assert score > 0.5


def test_stability_selection_close_to_all_features_baseline():
    """Single stochastic fit/seed, so allow a small tolerance rather than
    requiring the selected-feature score to strictly beat the all-features
    baseline every time (see `tests/basic_test.py` for the strict version
    at the library's other supported random_state).
    """
    breast = load_breast_cancer()
    X, y = breast.data, breast.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    orig_score = balanced_accuracy_score(y_test, clf.predict(X_test))

    gfs = AutoNFS(device=DEVICE, random_state=42)
    gfs.fit(X_train, y_train)
    clf.fit(gfs.transform(X_train), y_train)
    score = balanced_accuracy_score(y_test, clf.predict(gfs.transform(X_test)))

    assert score >= orig_score - 0.02
