from market_structure import validated_evidence
def fair_value_gaps(output):
    return [z for a in validated_evidence(output).values() for z in a["zones"] if "FVG" in z["kind"]]