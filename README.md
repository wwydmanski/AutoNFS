# AutoNFS

AutoNFS is a deep learning model that can be used to select the most important features from a given dataset. The model is based on the Gumbel-Sigmoid distribution.

## Adaptive `balance`

`balance` trades feature-selection aggressiveness against risk of mask collapse. Default is
`auto` (adaptive). Pass `balance="auto"` to back off automatically per-dataset if
`1.0` collapses or underperforms; see `autonfs/adaptive.py` and the HPO study report/addendum.

## Benchmark: AutoNFS vs. standard feature-selection methods

Benchmarked against 6 standard methods (mutual information, ANOVA F-test, L1-embedded, Random
Forest importance, mRMR, Boruta) on **38 real-world datasets** (5–10,000 features, 5 seeds each),
with every method given the same feature budget `k` that AutoNFS itself selected on that run — so
differences reflect selection quality, not different compression.

![AutoNFS rank distribution vs. 6 standard FS methods](docs/rank_distribution.png)

Each column is a method; dots are its per-dataset median rank (1 = best of 7) across 37 datasets
(`Ailerons` excluded — AutoNFS's mask collapsed there); the black diamond is the mean rank.

**AutoNFS has the best mean rank overall (2.64 of 7)**, ahead of RF importance (3.17) and
budget-matched Boruta (3.66), and is the single best method on 15/37 datasets — more than double
any competitor. The edge holds regardless of dataset size; see `AutoNFS_HPO_report_addendum.md`
(§7) for the full breakdown, including a timing benchmark where AutoNFS's selection cost stays
flat with feature count while RFE/mutual-information scale by 2–3 orders of magnitude.

Note: Boruta run to its own (unconstrained) convergence picks ~3x more features than AutoNFS and
wins 78% of seeds in that setting — but that's not a fair budget comparison, hence why it's
capped to the same `k` above.

## Installation
To install the package, you can use pip:
```bash
pip install autonfs
```

## Usage examples
### Basic usage
```python
from autonfs import AutoNFS
from sklearn.datasets import load_breast_cancer

breast = load_breast_cancer()
X = breast.data
y = breast.target

gfs = AutoNFS()
X = gfs.fit_transform(X, y)

print(gfs.support_)
print(gfs.scores_)
```

### Performance verification
```python
from autonfs import AutoNFS
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

DEVICE = "cpu"

breast = load_breast_cancer()
X = breast.data
y = breast.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.train(X_train, y_train)
orig_score = balanced_accuracy_score(y_test, clf.predict(X_test))

print(f"Original score: {orig_score:.3f}. Original features: {X.shape[1]}")
# Original score: 0.958. Original features: 30

gfs = AutoNFS(verbose=True, device=DEVICE)
gfs.fit(X_train, y_train)

X_transformed = gfs.transform(X_train)
X_test_transformed = gfs.transform(X_test)

clf.fit(X_transformed, y_train)
y_pred = clf.predict(X_test_transformed)
score = balanced_accuracy_score(y_test, y_pred)
logger.info(f"Score after feature selection: {score}. Selected features: {sum(gfs.support_)}")
# Score after feature selection: 0.958. Selected features: 3
```
