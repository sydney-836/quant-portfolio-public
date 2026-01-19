# execution/broker_alpaca.py
import os
import json
from typing import Dict, Any, Optional, List

import pandas as pd
from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame
from datetime import datetime, timedelta
import time as _time

# ✅ Single source of truth for persistence
from core.state_log import STATE_LOG, update_state


# ============================================================
# Env helpers
# ============================================================
def _env(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return default


def get_client():
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL") or os.getenv("APCA_API_BASE_URL")

    if not key or not secret or not base_url:
        raise RuntimeError("Missing Alpaca credentials")

    return REST(key, secret, base_url)


# ============================================================
# Runtime state helpers
# ============================================================
def _get_ns(ns: str) -> Dict[str, Any]:
    d = STATE_LOG.get(ns, {})
    return d if isinstance(d, dict) else {}


def _set_ns(ns: str, d: Dict[str, Any]) -> None:
    if isinstance(d, dict):
        update_state(ns, d)


def get_symbol_meta(symbol: str) -> Dict[str, Any]:
    sym = str(symbol).upper()
    store = _get_ns("symmeta")
    meta = store.get(sym, {}) if isinstance(store.get(sym, {}), dict) else {}
    meta.setdefault("layers", 1)
    meta.setdefault("last_entry_price", None)
    return meta


def set_symbol_meta(symbol: str, meta: Dict[str, Any]) -> None:
    if not isinstance(meta, dict):
        return
    sym = str(symbol).upper()
    store = _get_ns("symmeta")
    store[sym] = meta
    _set_ns("symmeta", store)


def get_trail_state(symbol: str) -> Dict[str, Any]:
    sym = str(symbol).upper()
    store = _get_ns("trail")
    st = store.get(sym, {}) if isinstance(store.get(sym, {}), dict) else {}
    st.setdefault("high_water", None)
    st.setdefault("last_stop", None)
    return st


def set_trail_state(symbol: str, state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    sym = str(symbol).upper()
    store = _get_ns("trail")
    store[sym] = state
    _set_ns("trail", store)


def clear_trail_state(symbol: str) -> None:
    sym = str(symbol).upper()
    store = _get_ns("trail")
    if sym in store:
        del store[sym]
        _set_ns("trail", store)


# ============================================================
# ATR computation
# ============================================================
def _compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or len(df) < period + 2:
        return None

    cols = {c.lower(): c for c in df.columns}
    if not all(k in cols for k in ("high", "low", "close")):
        return None

    high = df[cols["high"]].astype(float)
    low = df[cols["low"]].astype(float)
    close = df[cols["close"]].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(period).mean()
    val = atr.iloc[-1]
    return None if pd.isna(val) else float(val)


def enrich_atr(symbols: List[str], lookback_days: int = 90) -> Dict[str, Optional[float]]:
    out = {s: None for s in symbols}
    if not symbols:
        return out

    api = get_client()
    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days * 2)

    for symbol in symbols:
        try:
            bars = api.get_bars(
                symbol,
                TimeFrame.Day,
                start=start.isoformat(),
                end=end.isoformat(),
                feed="iex",
                adjustment="raw",
            )
            df = pd.DataFrame([{"high": b.h, "low": b.l, "close": b.c} for b in bars])
            out[symbol.upper()] = _compute_atr(df)
        except Exception:
            out[symbol.upper()] = None

    return out


# ============================================================
# Broker snapshot
# ============================================================
def read_only_check() -> Dict[str, Any]:
    api = get_client()
    acct = api.get_account()
    positions = api.list_positions()
    orders = api.list_orders(status="open")

    symbols = [p.symbol for p in positions]
    atr_map = enrich_atr(symbols)

    pos_dict = {}
    for p in positions:
        sym = p.symbol.upper()
        meta = get_symbol_meta(sym)

        pos_dict[sym] = {
            "qty": float(p.qty),
            "available_qty": float(getattr(p, "available_qty", 0.0)),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "atr": atr_map.get(sym),
            "layers": int(meta.get("layers", 1)),
            "last_entry_price": meta.get("last_entry_price"),
            "strategy": meta.get("strategy"),
        }

    return {
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "positions": pos_dict,
        "orders": [o._raw for o in orders],
    }


def get_last_trade_price(symbol: str) -> float:
    api = get_client()
    t = api.get_latest_trade(symbol.upper())
    return float(t.price)


# ============================================================
# ORDER + SLIPPAGE SUPPORT
# ============================================================
def get_order_fill(order_id: str) -> Dict[str, Any]:
    api = get_client()
    o = api.get_order(order_id)

    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    return {
        "id": getattr(o, "id", order_id),
        "symbol": getattr(o, "symbol", None),
        "side": getattr(o, "side", None),
        "status": getattr(o, "status", None),
        "filled_avg_price": _f(getattr(o, "filled_avg_price", None)),
        "filled_qty": _f(getattr(o, "filled_qty", None)),
        "submitted_at": str(getattr(o, "submitted_at", "")),
        "filled_at": str(getattr(o, "filled_at", "")),
    }


def submit_market_order_with_fill(
    symbol: str,
    qty: float,
    side: str,
    poll_seconds: float = 5.0,
    poll_interval: float = 0.5,
) -> Dict[str, Any]:
    api = get_client()
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy/sell")

    o = api.submit_order(
        symbol=symbol.upper(),
        qty=str(int(qty)),
        side=side,
        type="market",
        time_in_force="day",
    )

    oid = o.id
    deadline = _time.time() + poll_seconds
    last = {"id": oid, "status": o.status}

    while _time.time() < deadline:
        try:
            info = get_order_fill(oid)
            last = info
            if info.get("filled_avg_price") is not None:
                break
        except Exception:
            pass
        _time.sleep(poll_interval)

    return last


# ============================================================
# SAFE CLOSE (MINIMAL FIX)
# ============================================================
def close_position(symbol: str) -> None:
    api = get_client()

    try:
        pos = api.get_position(symbol.upper())
        available = float(getattr(pos, "available_qty", 0.0))
        if available <= 0:
            return
    except Exception:
        return

    api.close_position(symbol.upper())
