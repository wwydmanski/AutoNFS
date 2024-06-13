from sklearn.datasets import load_iris, load_breast_cancer
from gfs_network import GFSNetwork
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from logging import basicConfig, INFO, getLogger
import torch

basicConfig(level=INFO)
logger = getLogger(__name__)
DEVICE = "cpu"

def test_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target

    gfs = GFSNetwork(verbose=True, device=DEVICE)
    gfs.fit(X, y)
    print(gfs.support_)


def test_breast_cancer():
    breast = load_breast_cancer()
    X = breast.data
    y = breast.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    orig_score = balanced_accuracy_score(y_test, y_pred)
    logger.info(f"Original score: {orig_score:.3f}. Original features: {X.shape[1]}")

    gfs = GFSNetwork(verbose=True, device=DEVICE)
    gfs.fit(X_train, y_train)
    
    X_transformed = gfs.transform(X_train)
    X_test_transformed = gfs.transform(X_test)
    
    clf.fit(X_transformed, y_train)
    y_pred = clf.predict(X_test_transformed)
    score = balanced_accuracy_score(y_test, y_pred)
    logger.info(f"Score after feature selection: {score:.3f}. Selected features: {sum(gfs.support_)}")
    
    assert score >= orig_score
    

if __name__ == "__main__":
    logger.info("Iris Dataset")
    test_iris()

    logger.info("Breast Cancer Dataset")
    test_breast_cancer()