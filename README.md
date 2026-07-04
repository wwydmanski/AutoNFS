# AutoNFS

AutoNFS is a deep learning model that can be used to select the most important features from a given dataset. The model is based on the Gumbel-Sigmoid distribution.

## Adaptive `balance`

`balance` trades feature-selection aggressiveness against risk of mask collapse. Default is
`1.0` (max aggressiveness). Pass `balance="auto"` to back off automatically per-dataset if
`1.0` collapses or underperforms; see `autonfs/adaptive.py` and the HPO study report/addendum.

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
