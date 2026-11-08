import numpy as np
from market_structure import validated_evidence
FEATURE_NAMES = ("bull_score", "bear_score", "agreement", "atr", "adx", "efficiency", "atr_percentile", "rsi", "macd", "relative_volume", "ema_slope", "band_width", "htf", "ob_count", "fvg_count", "liquidity_count", "hour_ist", "minute_ist")
def features(output):
    frames = validated_evidence(output)
    r, s = output["report"], output["statistics"]
    a = frames["5m"]
    local_minutes = (output["asOf"] // 60000 + 330) % 1440
    return np.array([r["bull"]["score"], r["bear"]["score"], output["agreement"], s["atr"], s["adx"], s["efficiency"], s["atrPercentile"], s["rsi"], s["macd"], s.get("relativeVolume") if s.get("relativeVolume") is not None else np.nan, s["emaSlope"], s["bandWidth"], {"bull": 1, "bear": -1, "range": 0}[frames["1h"]["trend"]], sum("OB" in z["kind"] for z in a["zones"]), sum("FVG" in z["kind"] for z in a["zones"]), len(a["liquidity"]), local_minutes // 60, local_minutes % 60], dtype=float)