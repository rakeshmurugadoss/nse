def allowed_influence(registry):
    return bool(registry.get("calibrated") and registry.get("validated_override") and registry.get("beats_rule_baseline") and registry.get("beats_logistic_baseline") and registry.get("final_test_untouched"))
def predict(model, vector, registry):
    if not allowed_influence(registry):
        return {"score": None, "status": "Unvalidated", "may_override": False}
    return {"score": float(model.predict_proba([vector])[0, 1]), "status": "Calibrated", "may_override": True}