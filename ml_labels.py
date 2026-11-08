"""Versioned labels from real future bars; None means unknown, ambiguous or ineligible."""
from backtesting import barrier_label
LABEL_VERSION = "nse-labels-1"
TASKS = ("up_5m", "down_5m", "up_15m", "down_15m", "target_before_stop", "volatility_expansion", "continuation", "sweep_reversal")
def label(task, output, candles):
    if task not in TASKS:
        raise ValueError("Unsupported task")
    horizon = 5 if task in ("up_5m", "down_5m") else 15
    start = output["asOf"]
    future = [c for c in candles if start <= c["t"] < start + horizon * 60000]
    if len(future) != horizon or any(c["t"] != start + i * 60000 for i, c in enumerate(future)):
        return None
    price = output["report"]["price"]
    change = future[-1]["c"] - price
    # Threshold frozen at prediction time; no future volatility used.
    threshold = output["statistics"]["atr"] * .1
    if task.startswith("up_"): return int(change > threshold)
    if task.startswith("down_"): return int(change < -threshold)
    if task == "volatility_expansion":
        return int(max(c["h"] for c in future) - min(c["l"] for c in future) > output["statistics"]["atr"] * 2)
    if task == "target_before_stop":
        plan = output["report"].get("plan")
        if not plan or not plan.get("targets"): return None
        outcome = barrier_label(future, plan["targets"][0]["price"], plan["sl"], plan["dir"])
        return 1 if outcome == "target_first" else 0 if outcome == "stop_first" else None
    if task == "continuation":
        trend = output["report"]["tfs"]["1h"]["trend"]
        return None if trend == "range" else int(change > threshold if trend == "bull" else change < -threshold)
    sweeps = [d for d in output.get("detections", []) if d["type"] == "Liquidity sweep & rejection" and start - d["availableAt"] <= 900000]
    if not sweeps: return None
    return int(change > threshold if sweeps[-1]["direction"] == "bull" else change < -threshold)