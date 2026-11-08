from market_structure import validated_evidence
def liquidity(output):
    return [z for a in validated_evidence(output).values() for z in a["liquidity"]]