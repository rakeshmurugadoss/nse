"""Versioned, strict contracts. Timestamps are timezone-aware candle OPEN times."""
from datetime import datetime
from typing import Literal
import math
from pydantic import BaseModel, ConfigDict, Field, model_validator

Timeframe = Literal["1m", "5m", "15m", "60m"]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class CandleInput(StrictModel):
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid(self):
        if self.timestamp.tzinfo is None:
            raise ValueError("Candle timestamps must include timezone")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("Inconsistent OHLC bounds")
        return self

class Thresholds(StrictModel):
    strong_body: float = Field(default=.7, gt=0, le=1)
    rejection_wick: float = Field(default=.6, gt=0, le=1)
    doji_body: float = Field(default=.1, ge=0, le=.3)
    spinning_body: float = Field(default=.3, gt=0, le=.5)
    small_wick: float = Field(default=.05, ge=0, le=.2)
    wide_range: float = Field(default=1.5, gt=1)
    narrow_range: float = Field(default=.6, gt=0, lt=1)
    high_volume: float = Field(default=1.5, gt=1)
    low_volume: float = Field(default=.7, gt=0, lt=1)
    displacement_atr: float = Field(default=1.5, gt=0)
    exhaustion_atr: float = Field(default=2, gt=0)
    near_atr: float = Field(default=.5, gt=0)
    lookback: int = Field(default=20, ge=10, le=200)
    atr_period: int = Field(default=14, ge=2, le=100)
    rsi_period: int = Field(default=14, ge=2, le=100)
    weights: dict[str, float] = Field(default_factory=lambda: {
        "body_strength": 20, "wick_behaviour": 15, "volume_confirmation": 20,
        "atr_normalized_range": 15, "momentum": 15, "market_context": 15})

    @model_validator(mode="after")
    def valid_weights(self):
        if self.doji_body >= self.spinning_body:
            raise ValueError("doji_body must be smaller than spinning_body")
        names = {"body_strength", "wick_behaviour", "volume_confirmation", "atr_normalized_range", "momentum", "market_context"}
        if set(self.weights) != names or any(not math.isfinite(v) or v < 0 for v in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("Provide six nonnegative finite weights with positive total")
        return self

class StructuralEvidence(StrictModel):
    id: str
    kind: Literal["OB", "FVG", "swing_high", "swing_low", "support", "resistance"]
    timeframe: Timeframe
    bottom: float = Field(gt=0)
    top: float = Field(gt=0)
    available_at: datetime
    status_at: datetime
    active: bool = True

    @model_validator(mode="after")
    def valid_times(self):
        if self.available_at.tzinfo is None or self.status_at.tzinfo is None or self.status_at < self.available_at or self.top < self.bottom:
            raise ValueError("Invalid evidence bounds/timing")
        return self

class Evidence(StrictModel):
    code: str
    rule: str
    measurements: dict[str, float | int | str | None]

class Anatomy(StrictModel):
    open: float
    high: float
    low: float
    close: float
    absolute_body: float
    range: float
    upper_wick: float
    lower_wick: float
    body_to_range: float
    upper_wick_to_range: float
    lower_wick_to_range: float
    close_position: float | None
    gap_from_previous: float | None
    range_ratio_previous: float | None
    range_state: str
    relative_body: float | None

class Strength(StrictModel):
    body_strength: float
    buying_pressure_proxy: float | None
    selling_pressure_proxy: float | None
    upper_wick_rejection: float
    lower_wick_rejection: float
    momentum_expansion: float | None
    bullish_sequence: int
    bearish_sequence: int
    atr_normalized_range: float | None

class Volume(StrictModel):
    volume: float | None
    relative_volume: float | None
    volume_expansion: float | None
    volume_price_relationship: str

class ScoreComponent(StrictModel):
    score: float | None
    weight: float
    reason: str

class Quality(StrictModel):
    total: float
    components: dict[str, ScoreComponent]
    reasons: list[str]
    interpretation: str = "Heuristic candle evidence quality; not win probability or a trade signal"

class Context(StrictModel):
    previous_candle: dict[str, float | str] | None
    previous_windows: dict[str, dict[str, float | int | None]]
    ema9: float | None
    ema20: float | None
    ema50: float | None
    vwap: float | None
    atr: float | None
    rsi: float | None
    relative_volume: float | None
    recent_high: float | None
    recent_low: float | None
    confirmed_swing_high: float | None
    confirmed_swing_low: float | None
    nearby_support: list[str]
    nearby_resistance: list[str]
    existing_zones: list[str]
    ema_alignment: str
    interpretation: str

class CandleResult(StrictModel):
    timestamp: datetime
    decision_timestamp: datetime
    timeframe: Timeframe
    candle_direction: Literal["bullish", "bearish", "neutral"]
    anatomy: Anatomy
    strength_metrics: Strength
    volume_metrics: Volume
    detected_patterns: list[str]
    rejection_type: str | None
    displacement_type: str | None
    context: Context
    quality_score: Quality
    evidence: list[Evidence]
    warnings: list[str]

class SnapshotRequest(StrictModel):
    prediction_id: str
    timeframes: list[Timeframe] = Field(default_factory=lambda: ["1m", "5m", "15m", "60m"], min_length=1, max_length=4)
    thresholds: Thresholds = Field(default_factory=Thresholds)

class BatchResult(StrictModel):
    schema_version: str = "candle-intelligence-1"
    source: str
    as_of: datetime
    timeframes: dict[str, list[CandleResult]]
    warnings: list[str]