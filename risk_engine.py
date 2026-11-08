def costs(entry, exit_price, quantity, brokerage=20, slippage=.0005, charges=0):
    if min(entry, exit_price, quantity) <= 0 or min(brokerage, slippage, charges) < 0:
        raise ValueError("Invalid risk or cost values")
    return 2 * brokerage + (entry + exit_price) * quantity * slippage + charges
def excursion(candles, entry, direction):
    if not candles:
        return {"mae": None, "mfe": None}
    hi, lo = max(c["h"] for c in candles), min(c["l"] for c in candles)
    return {"mae": max(0, entry - lo) if direction == "BUY" else max(0, hi - entry), "mfe": max(0, hi - entry) if direction == "BUY" else max(0, entry - lo)}