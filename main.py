from uuid import UUID
from fastapi import FastAPI, Depends, HTTPException
from auth import require_user, UserStore
from contracts import AnalysisRequest, TrainingRequest, BacktestRequest
from prediction_orchestrator import enrich
from ob_protection import protection_evidence
from candle_intelligence.api import router as candle_router

app = FastAPI(title="Liquidity AI research service", version="1.0.0")
app.include_router(candle_router)

async def owned_prediction(store, prediction_id):
    try: UUID(prediction_id)
    except ValueError: raise HTTPException(422, "Invalid prediction ID")
    rows = await store.rows("predictions", f"id=eq.{prediction_id}&select=*")
    if not rows: raise HTTPException(404, "Prediction not found")
    return rows[0]

@app.post("/api/v1/predictions/analyze")
@app.post("/api/v1/predictions/multi-timeframe")
async def analyze(request: AnalysisRequest, store: UserStore = Depends(require_user)):
    # TS creates and stores the authoritative causal report before enrichment.
    row = await owned_prediction(store, request.prediction_id)
    return enrich(row["output"])

@app.get("/api/v1/predictions/history")
async def history(store: UserStore = Depends(require_user)):
    return await store.rows("predictions", "select=id,symbol,timeframe,prediction_timestamp,output&order=created_at.desc&limit=100")

@app.post("/api/v1/structure/order-block-protection")
async def order_block_protection(request: AnalysisRequest, store: UserStore = Depends(require_user)):
    row = await owned_prediction(store, request.prediction_id)
    return protection_evidence(row["output"])

@app.get("/api/v1/predictions/{prediction_id}")
async def detail(prediction_id: str, store: UserStore = Depends(require_user)):
    return await owned_prediction(store, prediction_id)

@app.get("/api/v1/models/status")
async def status(store: UserStore = Depends(require_user)):
    rows = await store.rows("model_registry", "select=*&order=created_at.desc&limit=100")
    return {"models": rows, "status": "Not trained" if not rows else "See model validation records"}

@app.get("/api/v1/models/performance")
async def model_performance(store: UserStore = Depends(require_user)):
    return await store.rows("model_evaluations", "select=*&limit=100")

@app.post("/api/v1/models/train")
async def train(request: TrainingRequest, store: UserStore = Depends(require_user)):
    rows = await store.rows("prediction_outcomes", "select=*&limit=1000")
    if len(rows) < request.minimum_samples:
        raise HTTPException(409, "Insufficient real labelled history; training is blocked")
    raise HTTPException(503, "Durable training worker not configured; no model was trained")

@app.post("/api/v1/backtest/run")
async def backtest(request: BacktestRequest, store: UserStore = Depends(require_user)):
    for prediction_id in request.predictions:
        await owned_prediction(store, prediction_id)
    raise HTTPException(503, "Durable replay worker not configured; no backtest was executed")

@app.get("/api/v1/backtest/results")
async def backtest_results(store: UserStore = Depends(require_user)):
    return await store.rows("backtest_runs", "select=*&order=created_at.desc&limit=100")