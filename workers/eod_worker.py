# workers/eod_worker.py

# ============================================================
# HARD PROOF LOGS (DO NOT REMOVE)
# ============================================================
print("EOD WORKER FILE LOADED ✅", flush=True)

import os
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Callable, Optional, Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from utils.proxy_cache import save_proxy_cache
from utils.universe_cache import save_universe

# SAFE MODULE IMPORT (NEVER FAILS)
import universe.expand as expand_module

# Persistence + state
from core.state_log import STATE_LOG, update_state

# If you still use these in your stack, keep them.
# If they don’t exist in your repo, this file will still run safely.
try:
    from core.metrics import finalize_day
except Exception:
    finalize_day = None

try:
    from core.state_store import init_db
except Exception:
    init_db = None


# ============================================================
# CONFIG
# ============================================================
ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")

INDEX_PROXY_SYMBOL = os.getenv("INDEX_PROXY_SYMBOL", "SPY")
PHASE3_MIN_SIZE = int(os.getenv("PHASE3_MIN_SIZE", "200"))

# Universe save gating (prevents same-day mutation)
# Default: only save the new universe after 22:00 UTC
UNIVERSE_SAVE_AFTER_HOUR_UTC = int(os.getenv("UNIVERSE_SAVE_AFTER_HOUR_UTC", "22"))

# Optional: store EOD snapshot under this key
EOD_STATE_KEY = os.getenv("EOD_STATE_KEY", "eod")


# ============================================================
# DATA CLIENT
# ============================================================
def _alpaca_data_client() -> StockHistoricalDataClient:
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing Alpaca data credentials (APCA_API_KEY_ID / APCA_API_SECRET_KEY)")
    return StockHistoricalDataClient(
        api_key=ALPACA_KEY,
        secret_key=ALPACA_SECRET,
    )


def fetch_price_history(
    symbols: List[str],
    lookback_days: int = 420,
) -> Dict[str, List[float]]:
    if not symbols:
        return {}

    client = _alpaca_data_client()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=list(dict.fromkeys([s.upper() for s in symbols if s])),
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    try:
        bars = client.get_stock_bars(request).df
    except Exception as e:
        print(f"EOD → price fetch failed: {e}", flush=True)
        return {}

    out: Dict[str, List[float]] = {}
    for sym in symbols:
        sym = sym.upper()
        try:
            if sym not in bars.index.get_level_values(0):
                continue
            closes = bars.xs(sym, level=0)["close"].dropna().tolist()
            if len(closes) >= 120:
                out[sym] = closes
        except Exception:
            continue

    return out


# ============================================================
# PROXY CACHE
# ============================================================
def refresh_proxy_cache(proxy_symbol: str) -> None:
    prices = fetch_price_history([proxy_symbol]).get(proxy_symbol.upper())

    if not prices or len(prices) < 200:
        print(f"PROXY_CACHE → FAILED {proxy_symbol}", flush=True)
        return

    save_proxy_cache(proxy_symbol.upper(), prices)
    print(f"PROXY_CACHE → updated {proxy_symbol} bars={len(prices)}", flush=True)


# ============================================================
# PHASE-3 BUILDER RESOLUTION (AUTHORITATIVE)
# ============================================================
def resolve_phase3_builder() -> Optional[Callable[[], List[str]]]:
    """
    Dynamically resolve the Phase-3 universe builder without assuming symbol names.
    """
    candidates = [
        "build_phase3_universe",
        "build_universe",
        "build_expanded_universe",
        "expand_universe",
    ]

    for name in candidates:
        fn = getattr(expand_module, name, None)
        if callable(fn):
            print(f"EOD → Phase-3 builder resolved: {name}", flush=True)
            return fn

    print("EOD → ❌ No valid Phase-3 universe builder found in universe.expand", flush=True)
    print("EOD → Available symbols:", [k for k in dir(expand_module) if k.startswith("build")], flush=True)
    return None


# ============================================================
# HELPERS
# ============================================================
def _now() -> datetime:
    return datetime.now(timezone.utc)

def _today_iso() -> str:
    return date.today().isoformat()

def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def should_save_universe_now() -> bool:
    """
    Prevents EOD from mutating the tradable universe during the trading day.
    Default behavior: only save after UNIVERSE_SAVE_AFTER_HOUR_UTC (UTC).
    """
    return _now().hour >= UNIVERSE_SAVE_AFTER_HOUR_UTC


def get_execution_snapshot() -> Dict[str, Any]:
    """
    EOD should NOT call broker APIs. Execution is the source of truth.
    """
    ex = STATE_LOG.get("execution", {}) or {}
    return {
        "equity": safe_float(ex.get("equity")),
        "cash": safe_float(ex.get("cash")),
        "gross_exposure": safe_float(ex.get("gross_exposure")),
        "net_exposure": safe_float(ex.get("net_exposure")),
        "cap_pressure": int(ex.get("cap_pressure", 0)),
        "positions": int(ex.get("positions", 0)),
        "status": ex.get("status", "unknown"),
        "ts": ex.get("ts") or ex.get("timestamp"),
    }


def get_allocator_governance() -> Dict[str, Any]:
    """
    Cleanly derive core vs satellite weights from allocator intent.
    Core = core_tech only. Everything else (continuation/momentum/MR/vol_sleeve) = satellite/other.
    """
    a = STATE_LOG.get("allocator", {}) or {}
    targets = a.get("targets", []) or []

    core_w = 0.0
    sat_w = 0.0
    other_w = 0.0

    for t in targets:
        s = str(t.get("strategy", "")).lower().strip()
        w = safe_float(t.get("target_exposure", 0.0))
        if s == "core_tech":
            core_w += w
        elif s in ("continuation", "momentum", "mean_reversion"):
            sat_w += w
        else:
            other_w += w  # vol_sleeve/hedge/unknown

    return {
        "regime": a.get("regime", "Unknown"),
        "confidence": safe_float(a.get("confidence", 0.0)),
        "target_gross": safe_float(a.get("target_gross", 0.0)),
        "core_weight": round(core_w, 4),
        "satellite_weight": round(sat_w, 4),
        "other_weight": round(other_w, 4),
        "universe_size": a.get("universe_size"),
        "satellite_universe_size": a.get("satellite_universe_size"),
    }


def summarize_trade_log() -> Dict[str, Any]:
    tl = STATE_LOG.get("trade_log", []) or []
    if not isinstance(tl, list) or not tl:
        return {
            "trades_logged": 0,
            "avg_slippage_bps": 0.0,
            "max_slippage_bps": 0.0,
            "min_slippage_bps": 0.0,
        }

    slips = []
    for r in tl:
        try:
            bps = float(r.get("slippage_bps"))
            slips.append(bps)
        except Exception:
            continue

    if not slips:
        return {
            "trades_logged": 0,
            "avg_slippage_bps": 0.0,
            "max_slippage_bps": 0.0,
            "min_slippage_bps": 0.0,
        }

    return {
        "trades_logged": int(len(slips)),
        "avg_slippage_bps": round(sum(slips) / len(slips), 2),
        "max_slippage_bps": round(max(slips), 2),
        "min_slippage_bps": round(min(slips), 2),
    }


# ============================================================
# MAIN
# ============================================================
def run_eod():
    print("EOD run_eod() ENTERED 🚀", flush=True)

    # --------------------------------------------------
    # EXECUTION READINESS GUARD (ONLY ADDITION)
    # --------------------------------------------------
    ex_guard = STATE_LOG.get("execution", {}) or {}
    if not ex_guard or safe_float(ex_guard.get("equity")) <= 0:
        print("EOD → execution state not ready, skipping snapshot", flush=True)
        return

    # Optional DB init if your stack uses it
    if callable(init_db):
        try:
            init_db()
        except Exception as e:
            print(f"EOD → init_db() failed (continuing): {e}", flush=True)

    # --------------------------------------------------
    # 1) PROXY CACHE (safe maintenance)
    # --------------------------------------------------
    try:
        refresh_proxy_cache(INDEX_PROXY_SYMBOL)
    except Exception as e:
        print(f"EOD → proxy cache refresh failed: {e}", flush=True)

    # --------------------------------------------------
    # 2) PHASE-3 UNIVERSE (safe maintenance, T+1 gated)
    # --------------------------------------------------
    if should_save_universe_now():
        builder = resolve_phase3_builder()
        if builder is None:
            print("EOD → Phase-3 skipped — no builder available", flush=True)
        else:
            try:
                phase3 = builder()
            except Exception as e:
                print(f"EOD → Phase-3 build FAILED: {e}", flush=True)
                phase3 = None

            if phase3 and len(phase3) >= PHASE3_MIN_SIZE:
                try:
                    save_universe(phase3)
                    update_state("phase3_universe_meta", {
                        "saved": True,
                        "count": int(len(phase3)),
                        "ts": _now().isoformat(),
                        "effective": "next_session",
                    })
                    print(f"EOD → Phase-3 SAVED ({len(phase3)}) — gated ✅", flush=True)
                except Exception as e:
                    print(f"EOD → Phase-3 save FAILED: {e}", flush=True)
            else:
                print("EOD → Phase-3 unavailable or too small — retaining prior universe", flush=True)
    else:
        print(
            f"EOD → Phase-3 save gated (hour_utc={_now().hour} < {UNIVERSE_SAVE_AFTER_HOUR_UTC})",
            flush=True
        )

    # --------------------------------------------------
    # 3) OBSERVABILITY SNAPSHOT (NO broker calls)
    # --------------------------------------------------
    ex = get_execution_snapshot()
    gov = get_allocator_governance()
    trade_stats = summarize_trade_log()

    result = None
    if callable(finalize_day):
        try:
            result = finalize_day(
                equity=ex["equity"],
                regime=gov["regime"],
                gross_exposure=ex["gross_exposure"],
                core_weight=gov["core_weight"],
                satellite_weight=gov["satellite_weight"],
                heat=safe_float((STATE_LOG.get("allocator", {}) or {}).get("risk", {}).get("heat", 0.0)),
                notes="EOD finalize (observability-only)",
            )
        except Exception as e:
            print(f"EOD → finalize_day failed (continuing): {e}", flush=True)
            result = None

    snapshot = {
        "date": _today_iso(),
        "ts": _now().isoformat(),

        "execution_status": ex["status"],
        "equity": round(ex["equity"], 2),
        "cash": round(ex["cash"], 2),
        "positions": ex["positions"],
        "gross_exposure": round(ex["gross_exposure"], 4),
        "net_exposure": round(ex["net_exposure"], 4),
        "cap_pressure": ex["cap_pressure"],

        "regime": gov["regime"],
        "confidence": round(gov["confidence"], 4),
        "target_gross": round(gov["target_gross"], 4),
        "core_weight": gov["core_weight"],
        "satellite_weight": gov["satellite_weight"],
        "other_weight": gov["other_weight"],

        **trade_stats,
    }

    if isinstance(result, dict):
        if "drawdown" in result:
            snapshot["drawdown"] = safe_float(result.get("drawdown"))
        if "kill_switch" in result:
            snapshot["kill_switch"] = bool(result.get("kill_switch"))

    update_state(EOD_STATE_KEY, snapshot)
    print(f"[EOD DONE] stored key={EOD_STATE_KEY} equity={snapshot['equity']:.2f}", flush=True)


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    run_eod()
