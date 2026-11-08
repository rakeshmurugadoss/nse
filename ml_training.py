import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
from probability_calibration import calibration

def purged_walk_forward(times, label_ends, minimum_train=500, test_size=100, embargo_ms=900000):
    times, label_ends = np.asarray(times), np.asarray(label_ends)
    if np.any(np.diff(times) <= 0) or np.any(label_ends <= times):
        raise ValueError("Strict chronological samples and positive horizons required")
    final_start = int(len(times) * 0.8)
    for start in range(minimum_train, final_start - test_size + 1, test_size):
        train = np.where((np.arange(len(times)) < start) & (label_ends < times[start] - embargo_ms))[0]
        if len(train) >= minimum_train:
            yield train, np.arange(start, start + test_size)

def baseline():
    return make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))

def chronological_calibrated_model(X, y, times, label_ends, boosted=False):
    """Fit models and sigmoid calibrators on disjoint pre-test chronological partitions."""
    from sklearn.frozen import FrozenEstimator
    X, y, times, label_ends = map(np.asarray, (X, y, times, label_ends))
    if len(y) < 1000 or np.any(np.diff(times) <= 0):
        raise ValueError("Sufficient strictly chronological data required")
    calibration_start, test_start = int(len(y)*.6), int(len(y)*.8)
    train = np.where(label_ends < times[calibration_start] - 900000)[0]
    calibrate = np.where((np.arange(len(y)) >= calibration_start) & (label_ends < times[test_start] - 900000))[0]
    test = np.arange(test_start, len(y))
    if any(len(index) < 100 or len(np.unique(y[index])) != 2 for index in (train, calibrate, test)):
        raise ValueError("Each purged partition needs both classes and 100 samples")
    estimator = make_pipeline(SimpleImputer(add_indicator=True), HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1)) if boosted else baseline()
    estimator.fit(X[train], y[train])
    model = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid").fit(X[calibrate], y[calibrate])
    return model, {"calibrated": True, "final_test_untouched": True, "final_test": metrics(y[test], model.predict_proba(X[test])[:,1]), "validated_override": False}

def override_gate(candidate, logistic, rules):
    """Independent final-test cost results are mandatory, not just classification accuracy."""
    keys = ("net_expectancy", "max_drawdown", "pr_auc", "brier")
    if not all(all(isinstance(r.get(k), (float, int)) for k in keys) for r in (candidate, logistic, rules)):
        return False
    return candidate["net_expectancy"] > max(0, logistic["net_expectancy"], rules["net_expectancy"]) and candidate["max_drawdown"] <= rules["max_drawdown"] and candidate["pr_auc"] >= logistic["pr_auc"] and candidate["brier"] <= logistic["brier"]

def train_baseline(X, y, times, label_ends):
    X, y = np.asarray(X), np.asarray(y)
    if len(y) < 1000 or len(np.unique(y)) != 2 or min(np.bincount(y.astype(int))) < 100:
        raise ValueError("Need at least 1000 real labelled samples and 100 examples per class")
    folds = []
    for train, test in purged_walk_forward(times, label_ends):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        model = baseline().fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
        folds.append(metrics(y[test], p))
    if not folds:
        raise ValueError("Not enough purged chronological validation folds")
    # Final untouched period is evaluated once after fixed baseline selection.
    start = int(len(y) * 0.8)
    train = np.where(np.asarray(label_ends) < times[start] - 900000)[0]
    model = baseline().fit(X[train], y[train])
    final = metrics(y[start:], model.predict_proba(X[start:])[:, 1])
    return model, {"walk_forward": folds, "final_test": final, "validated_override": False, "calibrated": False, "reason": "Rule comparison, cost evaluation and independent calibration are required before influence."}

def metrics(y, p):
    pred = p >= .5
    return {"precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)), "pr_auc": float(average_precision_score(y, p)), "confusion_matrix": confusion_matrix(y, pred).tolist(), **calibration(y, p)}