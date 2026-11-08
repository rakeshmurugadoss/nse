from typing import Literal
from pydantic import BaseModel, Field, model_validator

class Candle(BaseModel):
    t: int
    o: float = Field(gt=0)
    h: float = Field(gt=0)
    l: float = Field(gt=0)
    c: float = Field(gt=0)
    v: float = Field(ge=0)

    @model_validator(mode="after")
    def consistent(self):
        if self.t % 60000 or self.h < max(self.o, self.c) or self.l > min(self.o, self.c) or self.h < self.l:
            raise ValueError("Invalid minute OHLC")
        return self

class AnalysisRequest(BaseModel):
    prediction_id: str

class BacktestRequest(BaseModel):
    predictions: list[str] = Field(min_length=1, max_length=1000)
    brokerage_per_order: float = Field(default=20, ge=0)
    slippage_per_side: float = Field(default=0.0005, ge=0, le=0.05)
    other_charges: float = Field(default=0, ge=0)

class TrainingRequest(BaseModel):
    task: Literal["up_5m", "down_5m", "up_15m", "down_15m", "target_before_stop", "volatility_expansion", "continuation", "sweep_reversal"]
    minimum_samples: int = Field(default=1000, ge=1000)

FEATURE_VERSION = "causal-nse-1"