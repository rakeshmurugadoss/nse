"""Consume authoritative, timestamped TS detections; never guess zones in Python."""
def validated_evidence(output):
    as_of = output["asOf"]
    frames = output["report"]["tfs"]
    for frame in frames.values():
        for name in ("swings", "events", "zones", "liquidity"):
            for item in frame[name]:
                available = item.get("availableAt")
                if available is None or available > as_of:
                    raise ValueError("Missing or future detection availability")
    return frames