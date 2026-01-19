import os
import sys
import datetime
import traceback
from typing import Dict, Any, List

import numpy as np
from flask import Flask, render_template, jsonify

# ============================================================
# FORCE PROJECT ROOT ON PYTHON PATH
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ============================================================
# Imports
# ============================================================
from universe.discovery import discover_universe
from universe.expand import fetch_us_equities
from data.alpaca_history import get_price_history
from core.performance_store import init_db, load_equity_curve, load_spy_curve
from core.state_log import STATE_LOG, load_state
from utils.model_state_reader import get_latest_model_state

# Alpaca live positions
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import PositionSide

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)
init_db()
STATE_LOG.setdefault("regime_history", [])

# ============================================================
# Alpaca Client
# ============================================================
def get_alpaca_client():
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    paper = os.getenv("PAPER", "True").lower() == "true"
    if not api_key or not secret_key:
        return None
    return TradingClient(api_key, secret_key, paper=paper)

def get_live_positions():
    client = get_alpaca_client()
    if not client:
        return {}
    try:
        positions = client.get_all_positions()
        live = {}
        for p in positions:
            if p.side == PositionSide.LONG and float(p.qty) > 0:
                live[p.symbol] = {
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "avg_entry": float(p.avg_entry_price),
                    "upl": float(p.unrealized_pl),
                    "upl_pc": float(p.unrealized_plpc) * 100,
                }
        return live
    except Exception:
        return {}

# ============================================================
# Discovery Cache (RESTORED – FIXES CRASH)
# ============================================================
DISCOVERY_CACHE = {
    "day": None,
    "regime": None,
    "result": None,
}

def current_trading_day():
    return datetime.datetime.utcnow().date().isoformat()

def get_cached_discovery(market_regime: str):
    today = current_trading_day()

    if (
        DISCOVERY_CACHE["result"] is not None
        and DISCOVERY_CACHE["day"] == today
        and DISCOVERY_CACHE["regime"] == market_regime
    ):
        return DISCOVERY_CACHE["result"], today

    try:
        symbols = fetch_us_equities()[:200]
        prices = get_price_history(symbols, lookback_days=180)
        ranked = discover_universe(prices, market_regime)

        ranked.setdefault("continuation", [])
        ranked.setdefault("momentum", [])
        ranked.setdefault("mean_reversion", [])
        ranked["_using_fallback"] = False

    except Exception:
        ranked = {
            "continuation": [],
            "momentum": [],
            "mean_reversion": [],
            "_using_fallback": True,
        }

    DISCOVERY_CACHE["result"] = ranked
    DISCOVERY_CACHE["day"] = today
    DISCOVERY_CACHE["regime"] = market_regime

    return ranked, today

# ============================================================
# Risk & Analytics
# ============================================================
def compute_exposure_drift(gross_target: float) -> float:
    """
    How far current gross deviates from neutral 0.60
    """
    return round(abs(gross_target - 0.60), 3)

def compute_up_down_capture(strategy_eq: List[float], spy_eq: List[float]):
    if len(strategy_eq) < 2 or len(spy_eq) < 2:
        return None, None

    s = np.diff(strategy_eq) / strategy_eq[:-1]
    m = np.diff(spy_eq) / spy_eq[:-1]

    up_mask = m > 0
    down_mask = m < 0

    up_capture = np.sum(s[up_mask]) / np.sum(m[up_mask]) if np.any(up_mask) else None
    down_capture = np.sum(s[down_mask]) / np.sum(m[down_mask]) if np.any(down_mask) else None

    return (
        round(float(up_capture), 3) if up_capture is not None else None,
        round(float(abs(down_capture)), 3) if down_capture is not None else None,
    )

def compute_protection_score(downside_capture):
    if downside_capture is None:
        return None
    return round(max(0.0, 1.0 - downside_capture), 3)

def compute_heat(confidence: float, drawdown: float):
    heat = 0.5 + 0.75 * drawdown - 0.25 * (confidence - 0.5)
    return round(max(0.0, min(1.0, heat)), 3)

# ============================================================
# Routes
# ============================================================
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.route("/")
def dashboard():
    try:
        load_state()

        model_state = get_latest_model_state() or {}
        allocator_state = STATE_LOG.get("allocator", {}) or {}

        market_state = {
            "regime": model_state.get("regime", "Unknown"),
            "confidence": float(model_state.get("confidence", 0.0) or 0.0),
            "drawdown": float(model_state.get("drawdown", 0.0) or 0.0),
        }

        gross_target = float(
            allocator_state.get(
                "target_gross",
                model_state.get("target_gross", 0.0),
            ) or 0.0
        )

        allocator_targets = allocator_state.get("targets", []) or []
        live_positions = get_live_positions()

        equity_curve = load_equity_curve() or {}
        spy_curve = load_spy_curve() or {}

        strategy_eq = equity_curve.get("equity", []) or []
        spy_eq = spy_curve.get("equity", []) or []

        up_cap, down_cap = compute_up_down_capture(strategy_eq, spy_eq)
        protection = compute_protection_score(down_cap)
        exposure_drift = compute_exposure_drift(gross_target)

        ranked, discovery_day = get_cached_discovery(market_state["regime"])
        ranked_rows = (
            ranked.get("continuation", [])
            + ranked.get("momentum", [])
            + ranked.get("mean_reversion", [])
        )[:20]

        return render_template(
            "dashboard.html",
            timestamp=datetime.datetime.utcnow(),
            discovery_timestamp=discovery_day,
            market_state=market_state,
            gross_target=gross_target,
            allocator_targets=allocator_targets,
            live_positions=live_positions,
            ranked=ranked_rows,
            equity_curve=equity_curve,
            spy_curve=spy_curve,
            upside_capture=up_cap,
            downside_capture=down_cap,
            protection_score=protection,
            exposure_drift=exposure_drift,
            risk={"heat": compute_heat(market_state["confidence"], market_state["drawdown"])},
            regime_history=STATE_LOG.get("regime_history", []),
        )

    except Exception:
        traceback.print_exc()
        return "Dashboard error — check logs", 500

# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
