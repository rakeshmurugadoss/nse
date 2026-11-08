from risk_engine import costs
def barrier_label(candles, target, stop, direction):
    for c in candles:
        target_hit = c["h"] >= target if direction == "BUY" else c["l"] <= target
        stop_hit = c["l"] <= stop if direction == "BUY" else c["h"] >= stop
        if target_hit and stop_hit:
            return "ambiguous"
        if target_hit:
            return "target_first"
        if stop_hit:
            return "stop_first"
    return "neither"
def performance(trades):
    equity = peak = drawdown = 0
    pnls = []
    for t in trades:
        gross = (t["exit"] - t["entry"]) * t["quantity"] * (1 if t["direction"] == "BUY" else -1)
        pnl = gross - costs(t["entry"], t["exit"], t["quantity"], t.get("brokerage", 20), t.get("slippage", .0005), t.get("charges", 0))
        pnls.append(pnl); equity += pnl; peak = max(peak, equity); drawdown = max(drawdown, peak - equity)
    wins, losses = sum(max(p, 0) for p in pnls), -sum(min(p, 0) for p in pnls)
    return {"count": len(pnls), "net": equity, "net_expectancy": equity / len(pnls) if pnls else None, "max_drawdown": drawdown, "profit_factor": wins / losses if losses else None, "sharpe": None, "sharpe_reason": "Requires a defined return series, capital and sampling frequency"}