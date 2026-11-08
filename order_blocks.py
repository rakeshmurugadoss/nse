from market_structure import validated_evidence
def order_blocks(output):
    return [z for a in validated_evidence(output).values() for z in a["zones"] if "OB" in z["kind"] or "Breaker" in z["kind"]]