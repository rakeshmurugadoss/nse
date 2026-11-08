"""Owner-authenticated, read-only analysis of existing stored real candle snapshots."""
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool
from auth import UserStore, require_user
from .schemas import SnapshotRequest, BatchResult
from .engine import analyze_minutes
from .snapshot import snapshot_inputs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/candles", tags=["Candle intelligence"])

@router.post("/intelligence", response_model=BatchResult)
async def intelligence(request: SnapshotRequest, store: UserStore = Depends(require_user)):
    try:
        UUID(request.prediction_id)
    except ValueError:
        raise HTTPException(422, "Invalid prediction ID")
    rows = await store.rows("predictions", f"id=eq.{request.prediction_id}&select=output,snapshot_hash,data_source")
    if not rows:
        raise HTTPException(404, "Prediction not found")
    output = rows[0].get("output", {})
    if not output.get("candles"):
        raise HTTPException(409, "Saved prediction has no OHLCV snapshot")
    if len(output["candles"]) > 10000:
        raise HTTPException(413, "Snapshot exceeds 10,000 minute candles; select a shorter research period")
    try:
        candles, as_of, structures = snapshot_inputs(output)
        result = await run_in_threadpool(analyze_minutes, candles, as_of, request.timeframes, request.thresholds, structures, "owned_prediction_snapshot")
        logger.info("Candle intelligence completed: rows=%d frames=%d",len(candles),len(result.timeframes))
        return result
    except (ValueError, KeyError, TypeError):
        logger.warning("Invalid saved candle snapshot")
        raise HTTPException(422, "Saved candle snapshot or structural evidence is invalid")