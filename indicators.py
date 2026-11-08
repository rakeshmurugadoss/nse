from market_structure import validated_evidence
def indicators(output):
    return {tf: {k: a.get(k) for k in ("atr", "rsi", "ema9", "ema20", "ema50", "vwap", "relVol")} for tf, a in validated_evidence(output).items()}