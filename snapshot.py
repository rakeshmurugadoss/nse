"""Adapters for immutable TS snapshots; structural state is never redetected."""
from datetime import datetime, timezone
from .schemas import CandleInput, StructuralEvidence

def ms(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc)

def snapshot_inputs(output):
    candles = [CandleInput(timestamp=ms(c["t"]),open=c["o"],high=c["h"],low=c["l"],close=c["c"],volume=c.get("v")) for c in output["candles"]]
    structures = []
    frames = output.get("report", {}).get("tfs", output.get("report", {}).get("frames", []))
    if isinstance(frames, dict):
        frames = list(frames.values())
    for frame in frames:
        tf = "60m" if frame.get("tf") == "1h" else frame.get("tf")
        if tf not in ("1m","5m","15m","60m"):
            continue
        for j, z in enumerate(frame.get("zones", [])):
            if not z.get("availableAt") or not z.get("statusAt"):
                continue
            kind = "OB" if "OB" in z["kind"] else "FVG" if "FVG" in z["kind"] else None
            if kind:
                structures.append(StructuralEvidence(id=f"{tf}:zone:{j}",kind=kind,timeframe=tf,
                    bottom=z["bottom"],top=z["top"],available_at=ms(z["availableAt"]),status_at=ms(z["statusAt"]),
                    active=z.get("status") in ("Untouched","Reclaimed")))
        for j,s in enumerate(frame.get("swings", [])):
            if s.get("availableAt"):
                # Pivots are immutable: use confirmation time, not future final-frame status.
                structures.append(StructuralEvidence(id=f"{tf}:swing:{j}",kind="swing_high" if s["type"]=="high" else "swing_low",
                    timeframe=tf,bottom=s["price"],top=s["price"],available_at=ms(s["availableAt"]),status_at=ms(s["availableAt"])))
        for j,l in enumerate(frame.get("liquidity", [])):
            if l.get("availableAt") and l.get("statusAt"):
                structures.append(StructuralEvidence(id=f"{tf}:liquidity:{j}",kind="resistance" if l["kind"] in ("BSL","EQH","PDH") else "support",
                    timeframe=tf,bottom=l["price"],top=l["price"],available_at=ms(l["availableAt"]),status_at=ms(l["statusAt"]),active=l.get("status")=="Untouched"))
    return candles, ms(output["asOf"]), structures