"""Consume authoritative saved TypeScript protection evidence; never redetect zones."""
import math
from fastapi import HTTPException

VERSION = "shared-ob-protection-1"
ACTIVE = {"fresh", "touched", "partially_mitigated"}
STATES = ACTIVE | {"fully_mitigated", "invalidated", "expired"}

def protection_evidence(output):
    report = output.get("protection")
    if report is None:
        raise HTTPException(409, "This snapshot predates order-block protection; create a new analysis")
    def number(n):
        return isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n)
    as_of = output.get("asOf")
    if not isinstance(report, dict) or report.get("engineVersion") != VERSION or not number(as_of) or report.get("snapshotAt") != as_of:
        raise HTTPException(422, "Unsupported or invalid protection evidence")
    blocks = report.get("orderBlocks")
    if not isinstance(blocks, list):
        raise HTTPException(422, "Invalid order-block evidence")
    for b in blocks:
        if not isinstance(b, dict):
            raise HTTPException(422, "Invalid order-block evidence")
        times = [b.get(k) for k in ("formedAt", "confirmedAt", "availableAt", "statusAt")]
        if not all(number(t) for t in times) or not times[0] <= times[1] == times[2] <= times[3] == as_of:
            raise HTTPException(422, "Non-causal order-block evidence")
        if b.get("direction") not in {"bull", "bear"} or b.get("lifecycle") not in STATES or not isinstance(b.get("qualityPassed"), bool):
            raise HTTPException(422, "Invalid order-block lifecycle")
        if not number(b.get("bottom")) or not number(b.get("top")) or not 0 < b["bottom"] < b["top"]:
            raise HTTPException(422, "Invalid order-block bounds")
        if not number(b.get("qualityScore")) or not 0 <= b["qualityScore"] <= 100 or not number(b.get("mitigationFraction")) or not 0 <= b["mitigationFraction"] <= 1:
            raise HTTPException(422, "Invalid heuristic evidence")
        for key in ("firstTouchAt", "invalidatedAt", "expiredAt"):
            t = b.get(key)
            if t is not None and (not number(t) or not b["availableAt"] < t <= as_of):
                raise HTTPException(422, "Non-causal lifecycle timestamp")
        if b["lifecycle"] == "invalidated" and b.get("invalidatedAt") is None or b["lifecycle"] == "expired" and b.get("expiredAt") is None:
            raise HTTPException(422, "Missing lifecycle timestamp")
    # Do not trust a separately supplied active list: derive it from validated evidence.
    return {**report, "protectedActiveOrderBlocks": [b for b in blocks if b["qualityPassed"] and b["lifecycle"] in ACTIVE]}