from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
def calibration(y, probabilities):
    fraction, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
    return {"brier": float(brier_score_loss(y, probabilities)), "bins": [{"predicted": float(p), "observed": float(f)} for p, f in zip(predicted, fraction)]}