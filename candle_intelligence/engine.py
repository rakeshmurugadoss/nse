"""Deterministic causal candle features, independent of the authoritative zone engine.

No broker IO, signals, persistence or future-confirmed pivots. Input is real one-minute
OHLCV. NSE/BSE regular sessions are anchored at 09:15 Asia/Kolkata; the last 60m
bucket ends at 15:30. Baselines exclude the current candle; indicators include it
only after close. EMA is seeded by its first complete period; ATR/RSI use Wilder.
"""
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict
import numpy as np
import pandas as pd
from .schemas import (CandleInput, CandleResult, Thresholds, StructuralEvidence,
                      BatchResult, Timeframe)

IST = ZoneInfo("Asia/Kolkata")
MINUTES = {"1m": 1, "5m": 5, "15m": 15, "60m": 60}

def ratio(a, b):
    return float(a / b) if b is not None and b > 0 else None

def aggregate_minutes(candles: list[CandleInput], timeframe: Timeframe, as_of: datetime):
    """Require every one-minute slot; never fill gaps or emit forming buckets."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must include timezone")
    if any(b.timestamp <= a.timestamp for a, b in zip(candles, candles[1:])):
        raise ValueError("Candles must be strictly chronological and unique")
    buckets = defaultdict(list)
    warnings = []
    for c in candles:
        t = c.timestamp.astimezone(IST)
        if t.second or t.microsecond:
            raise ValueError("One-minute open timestamps must be minute aligned")
        if t.weekday() >= 5 or not time(9, 15) <= t.time() < time(15, 30):
            warnings.append("Excluded candle outside regular Indian-equity session")
            continue
        start = t.replace(hour=9, minute=15)
        offset = int((t - start).total_seconds() // 60)
        bucket = start + timedelta(minutes=offset // MINUTES[timeframe] * MINUTES[timeframe])
        buckets[bucket].append(c)
    result = []
    for start, rows in sorted(buckets.items()):
        end = min(start + timedelta(minutes=MINUTES[timeframe]), start.replace(hour=15, minute=30))
        expected = int((end - start).total_seconds() // 60)
        if end > as_of:
            warnings.append(f"Excluded forming {timeframe} bucket at {start.isoformat()}")
            continue
        if len(rows) != expected or any(r.timestamp != start + timedelta(minutes=i) for i, r in enumerate(rows)):
            warnings.append(f"Excluded incomplete {timeframe} bucket at {start.isoformat()}")
            continue
        result.append((CandleInput(timestamp=start, open=rows[0].open,
            high=max(r.high for r in rows), low=min(r.low for r in rows), close=rows[-1].close,
            volume=sum(r.volume for r in rows) if all(r.volume is not None for r in rows) else None), end))
    return result, list(dict.fromkeys(warnings))

def indicators(rows, cfg):
    """Vectorized causal indicators; null through warm-up and missing-volume sessions."""
    d = pd.DataFrame([r.model_dump() for r, _ in rows])
    if d.empty:
        return {}
    close = d.close
    result = {}
    for period in (9, 20, 50):
        vals = [None] * len(d)
        if len(d) >= period:
            vals[period - 1] = float(close.iloc[:period].mean())
            for i in range(period, len(d)):
                vals[i] = float(close.iloc[i] * (2 / (period + 1)) + vals[i - 1] * (1 - 2 / (period + 1)))
        result[f"ema{period}"] = vals
    tr = pd.concat([d.high - d.low, (d.high - close.shift()).abs(), (d.low - close.shift()).abs()], axis=1).max(axis=1)
    def wilder(series, period):
        vals = [None] * len(d)
        if len(d) >= period:
            vals[period - 1] = float(series.iloc[:period].mean())
            for i in range(period, len(d)):
                vals[i] = float((vals[i - 1] * (period - 1) + series.iloc[i]) / period)
        return vals
    result["atr"] = wilder(tr, cfg.atr_period)
    delta = close.diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    rsis = [None] * len(d)
    p = cfg.rsi_period
    if len(d) > p:
        gain, loss = float(gains.iloc[1:p+1].mean()), float(losses.iloc[1:p+1].mean())
        for i in range(p, len(d)):
            if i > p:
                gain = (gain * (p-1) + gains.iloc[i]) / p
                loss = (loss * (p-1) + losses.iloc[i]) / p
            rsis[i] = float(50 if gain == loss == 0 else 100 if loss == 0 else 100 - 100 / (1 + gain / loss))
    result["rsi"] = rsis
    vwaps, day, pv, vol, missing = [], None, 0, 0, False
    for r, _ in rows:
        current = r.timestamp.astimezone(IST).date()
        if day != current:
            day, pv, vol, missing = current, 0, 0, False
        if r.volume is None:
            missing = True
        else:
            pv += (r.high + r.low + r.close) / 3 * r.volume
            vol += r.volume
        vwaps.append(None if missing else ratio(pv, vol))
    result["vwap"] = vwaps
    return result

def analyze_frame(rows, timeframe, cfg, structures=()):
    """Properties and context stay separate; all detections carry exact measurements."""
    ind = indicators(rows, cfg)
    output, bull_seq, bear_seq = [], 0, 0
    for i, (c, decision) in enumerate(rows):
        past = [r for r, _ in rows[max(0, i-cfg.lookback):i]]
        prev = past[-1] if past else None
        body, span = abs(c.close-c.open), c.high-c.low
        upper, lower = c.high-max(c.open,c.close), min(c.open,c.close)-c.low
        br, ur, lr = (body/span, upper/span, lower/span) if span else (0, 0, 0)
        cp = ratio(c.close-c.low, span)
        direction = "bullish" if c.close > c.open else "bearish" if c.close < c.open else "neutral"
        bull_seq = bull_seq + 1 if direction == "bullish" else 0
        bear_seq = bear_seq + 1 if direction == "bearish" else 0
        baseline_ready = len(past) == cfg.lookback
        avg_range = float(np.mean([r.high-r.low for r in past])) if baseline_ready else None
        avg_body = float(np.mean([abs(r.close-r.open) for r in past])) if baseline_ready else None
        volumes = [r.volume for r in past]
        avg_vol = float(np.mean(volumes)) if baseline_ready and all(v is not None for v in volumes) else None
        relvol = ratio(c.volume, avg_vol) if c.volume is not None else None
        rr, rb = ratio(span, avg_range), ratio(body, avg_body)
        atr_prior = ind["atr"][i-1] if i else None
        atrnorm = ratio(span, atr_prior)
        recent_high = max((r.high for r in past), default=None)
        recent_low = min((r.low for r in past), default=None)
        evidence, patterns = [], []
        measurements = {"body_ratio": br, "upper_wick_ratio": ur, "lower_wick_ratio": lr,
            "close_position": cp, "range_baseline_ratio": rr, "relative_volume": relvol,
            "range_prior_atr": atrnorm, "high": c.high, "low": c.low, "close": c.close,
            "prior_high": recent_high, "prior_low": recent_low}
        def detect(code, condition, rule, pattern=False):
            if condition:
                evidence.append({"code": code, "rule": rule, "measurements": measurements.copy()})
                if pattern:
                    patterns.append(code)
        strongbull = direction == "bullish" and br >= cfg.strong_body
        strongbear = direction == "bearish" and br >= cfg.strong_body
        rejectbull = span > 0 and lr >= cfg.rejection_wick and cp >= .6
        rejectbear = span > 0 and ur >= cfg.rejection_wick and cp <= .4
        doji = span > 0 and br <= cfg.doji_body
        detect("strong_bullish", strongbull, f"close>open AND body/range>={cfg.strong_body}", True)
        detect("strong_bearish", strongbear, f"close<open AND body/range>={cfg.strong_body}", True)
        detect("bullish_rejection", rejectbull, f"lower_wick/range>={cfg.rejection_wick} AND close_position>=0.6", True)
        detect("bearish_rejection", rejectbear, f"upper_wick/range>={cfg.rejection_wick} AND close_position<=0.4", True)
        detect("long_legged_doji", doji and ur >= .35 and lr >= .35, f"body/range<={cfg.doji_body} AND both wick ratios>=0.35", True)
        detect("dragonfly_doji", doji and ur <= cfg.small_wick and lr >= .6, f"doji AND upper_wick/range<={cfg.small_wick} AND lower>=0.6", True)
        detect("gravestone_doji", doji and lr <= cfg.small_wick and ur >= .6, f"doji AND lower_wick/range<={cfg.small_wick} AND upper>=0.6", True)
        detect("spinning_top", cfg.doji_body < br <= cfg.spinning_body and ur >= .2 and lr >= .2, f"{cfg.doji_body}<body/range<={cfg.spinning_body} AND both wicks>=0.2", True)
        detect("marubozu", span > 0 and ur <= cfg.small_wick and lr <= cfg.small_wick, f"nonzero range AND both wick ratios<={cfg.small_wick}", True)
        detect("inside_bar", prev is not None and c.high < prev.high and c.low > prev.low, "high<previous high AND low>previous low (strict)", True)
        detect("outside_bar", prev is not None and c.high > prev.high and c.low < prev.low, "high>previous high AND low<previous low (strict)", True)
        detect("wide_range_bar", rr is not None and rr >= cfg.wide_range, f"range/prior {cfg.lookback}-bar mean range>={cfg.wide_range}", True)
        detect("narrow_range_bar", rr is not None and rr <= cfg.narrow_range, f"range/prior {cfg.lookback}-bar mean range<={cfg.narrow_range}", True)
        detect("high_volume_expansion", relvol is not None and relvol >= cfg.high_volume and rr is not None and rr >= cfg.wide_range, f"relative volume>={cfg.high_volume} AND range ratio>={cfg.wide_range}", True)
        detect("low_volume_compression", relvol is not None and relvol <= cfg.low_volume and rr is not None and rr <= cfg.narrow_range, f"relative volume<={cfg.low_volume} AND range ratio<={cfg.narrow_range}", True)
        displacement = "bullish" if strongbull and atrnorm is not None and atrnorm >= cfg.displacement_atr else "bearish" if strongbear and atrnorm is not None and atrnorm >= cfg.displacement_atr else None
        detect(f"strong_{displacement}_displacement", displacement is not None, f"directional strong body AND range/prior Wilder ATR>={cfg.displacement_atr}")
        sweep_up = baseline_ready and c.high > recent_high and c.close <= recent_high
        sweep_down = baseline_ready and c.low < recent_low and c.close >= recent_low
        detect("potential_liquidity_sweep_high", sweep_up, f"high>prior {cfg.lookback}-bar high AND close<=that high; price proxy only")
        detect("potential_liquidity_sweep_low", sweep_down, f"low<prior {cfg.lookback}-bar low AND close>=that low; price proxy only")
        detect("potential_breakout_up", baseline_ready and c.close > recent_high, f"close>prior {cfg.lookback}-bar high")
        detect("potential_breakout_down", baseline_ready and c.close < recent_low, f"close<prior {cfg.lookback}-bar low")
        # Failure is attached to today's closed candle, never relabelled on the breakout candle.
        before_prev = [r for r, _ in rows[max(0,i-1-cfg.lookback):i-1]]
        if prev and len(before_prev) == cfg.lookback:
            hi, lo = max(r.high for r in before_prev), min(r.low for r in before_prev)
            failed = (prev.close > hi and c.close <= hi) or (prev.close < lo and c.close >= lo)
            detect("failed_breakout", failed, f"previous close broke its own prior {cfg.lookback}-bar extreme AND current close returned inside")
            if failed:
                evidence[-1]["measurements"].update({"breakout_reference_high": hi, "breakout_reference_low": lo, "previous_close": prev.close})
        detect("potential_exhaustion", atrnorm is not None and atrnorm >= cfg.exhaustion_atr and max(ur,lr) >= cfg.rejection_wick, f"range/prior ATR>={cfg.exhaustion_atr} AND largest wick ratio>={cfg.rejection_wick}; not confirmed reversal")
        # Snapshot state cannot be projected backwards: exact decision-time state only.
        visible = [s for s in structures if s.timeframe == timeframe and s.available_at <= decision and s.active and
                   (s.status_at == decision or s.kind in ("swing_high", "swing_low") and s.status_at <= decision)]
        visible.sort(key=lambda s: s.available_at, reverse=True)
        nearby = [s for s in visible if (atr_prior is not None and min(abs(c.close-s.bottom),abs(c.close-s.top)) <= cfg.near_atr*atr_prior)
                  or s.bottom <= c.close <= s.top]
        e9,e20,e50 = (ind[f"ema{p}"][i] for p in (9,20,50))
        alignment = "unavailable" if e50 is None else "bullish" if e9>e20>e50 else "bearish" if e9<e20<e50 else "mixed"
        momentum = ratio(body, abs(prev.close-prev.open)) if prev else None
        windows = {}
        for n in (3,5,10):
            history = [r for r,_ in rows[max(0,i-n):i]]
            windows[str(n)] = {"count":len(history), "high":max(r.high for r in history) if len(history)==n else None,
                "low":min(r.low for r in history) if len(history)==n else None,
                "net_change":history[-1].close-history[0].open if len(history)==n else None}
        warnings = ["OHLCV pressure, sweep and exhaustion readings are proxies, not participant identities or orders"]
        if not span: warnings.append("Zero-range candle: ratios are zero and close position is unavailable")
        if not baseline_ready: warnings.append("Insufficient prior history for range/body/volume baselines")
        if relvol is None: warnings.append("No usable volume baseline; no volume confirmation credit")
        if e50 is None: warnings.append("EMA50 warming up; market-context score unavailable")
        if not visible: warnings.append("No authoritative structural state supplied at this candle decision timestamp")
        # Missing dimensions earn zero contribution; weights never re-normalize around missing data.
        scores = {"body_strength":br*100, "wick_behaviour":max(ur,lr)*100 if rejectbull or rejectbear else (1-ur-lr)*100,
            "volume_confirmation":min(100,relvol/cfg.high_volume*100) if relvol is not None else None,
            "atr_normalized_range":min(100,atrnorm/cfg.displacement_atr*100) if atrnorm is not None else None,
            "momentum":min(100,(rb or 0)/cfg.wide_range*100) if rb is not None else None,
            "market_context":100 if alignment==direction and direction!="neutral" else 0 if alignment=="unavailable" else 50}
        if alignment=="unavailable": scores["market_context"]=None
        reasons = {"body_strength":f"body/range={br:.6f}", "wick_behaviour":f"rejection={rejectbull or rejectbear}; upper={ur:.6f}, lower={lr:.6f}",
            "volume_confirmation":f"relative volume={relvol}; saturation={cfg.high_volume}", "atr_normalized_range":f"range/prior ATR={atrnorm}; saturation={cfg.displacement_atr}",
            "momentum":f"body/prior mean body={rb}; saturation={cfg.wide_range}", "market_context":f"direction={direction}; EMA9/20/50 alignment={alignment}"}
        total = sum((scores[k] or 0)*cfg.weights[k] for k in scores)/sum(cfg.weights.values())
        vol_state = "unavailable" if relvol is None else f"{'expanding' if relvol>=cfg.high_volume else 'contracting' if relvol<=cfg.low_volume else 'normal'} volume with {direction} price"
        output.append(CandleResult(timestamp=c.timestamp, decision_timestamp=decision, timeframe=timeframe,
            candle_direction=direction, anatomy={"open":c.open,"high":c.high,"low":c.low,"close":c.close,"absolute_body":body,"range":span,
                "upper_wick":upper,"lower_wick":lower,"body_to_range":br,"upper_wick_to_range":ur,"lower_wick_to_range":lr,"close_position":cp,
                "gap_from_previous":c.open-prev.close if prev else None,"range_ratio_previous":ratio(span,prev.high-prev.low) if prev else None,
                "range_state":"unavailable" if prev is None else "expansion" if span>prev.high-prev.low else "contraction" if span<prev.high-prev.low else "unchanged","relative_body":rb},
            strength_metrics={"body_strength":br,"buying_pressure_proxy":cp,"selling_pressure_proxy":1-cp if cp is not None else None,
                "upper_wick_rejection":ur,"lower_wick_rejection":lr,"momentum_expansion":momentum,"bullish_sequence":bull_seq,"bearish_sequence":bear_seq,"atr_normalized_range":atrnorm},
            volume_metrics={"volume":c.volume,"relative_volume":relvol,"volume_expansion":ratio(c.volume,prev.volume) if prev and c.volume is not None else None,"volume_price_relationship":vol_state},
            detected_patterns=patterns,rejection_type="both" if rejectbull and rejectbear else "bullish" if rejectbull else "bearish" if rejectbear else None,
            displacement_type=displacement, context={"previous_candle":prev.model_dump(exclude={"timestamp","volume"}) if prev else None,"previous_windows":windows,
                "ema9":e9,"ema20":e20,"ema50":e50,"vwap":ind["vwap"][i],"atr":ind["atr"][i],"rsi":ind["rsi"][i],"relative_volume":relvol,
                "recent_high":recent_high,"recent_low":recent_low,"confirmed_swing_high":next((s.top for s in visible if s.kind=="swing_high"),None),
                "confirmed_swing_low":next((s.bottom for s in visible if s.kind=="swing_low"),None),"nearby_support":[s.id for s in nearby if s.kind=="support"],
                "nearby_resistance":[s.id for s in nearby if s.kind=="resistance"],"existing_zones":[s.id for s in nearby if s.kind in ("OB","FVG")],
                "ema_alignment":alignment,"interpretation":"Recent extremes are trailing reference levels, not independently detected confirmed swings"},
            quality_score={"total":round(total,4),"components":{k:{"score":v,"weight":cfg.weights[k],"reason":reasons[k]} for k,v in scores.items()},
                "reasons":["Weighted sum divided by all configured weights; missing evidence contributes zero",*reasons.values()]}, evidence=evidence,warnings=warnings))
    return output

def analyze_minutes(candles: list[CandleInput], as_of: datetime, timeframes=("1m","5m","15m","60m"), thresholds=None, structures=(), source="existing_ohlcv") -> BatchResult:
    cfg = thresholds or Thresholds()
    frames, warnings = {}, []
    for tf in dict.fromkeys(timeframes):
        rows, excluded = aggregate_minutes(candles, tf, as_of)
        frames[tf] = analyze_frame(rows, tf, cfg, structures)
        warnings.extend(excluded)
    return BatchResult(source=source,as_of=as_of,timeframes=frames,warnings=list(dict.fromkeys(warnings)))