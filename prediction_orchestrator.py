from market_structure import validated_evidence
from order_blocks import order_blocks
from fair_value_gaps import fair_value_gaps
from liquidity import liquidity
from regime_detection import regime
from ml_features import features
from ob_protection import protection_evidence
def enrich(output):
    validated_evidence(output)
    return {"prediction": output, "regime": regime(output), "order_blocks": order_blocks(output), "protection": protection_evidence(output) if output.get("protection") is not None else None, "fair_value_gaps": fair_value_gaps(output), "liquidity": liquidity(output), "feature_count": len(features(output)), "model": {"status": "Not trained", "score": None, "validated": False}}