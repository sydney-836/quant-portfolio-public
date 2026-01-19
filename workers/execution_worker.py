# workers/execution_worker.py

import math
import os
from typing import Dict, Any, List
import time as _time
from datetime import datetime, timezone, date, timedelta, time as dt_time




from core.state_log import init_runtime_db, update_state, STATE_LOG
init_runtime_db()

from execution.broker_alpaca import (
    read_only_check,
    close_position,
    get_last_trade_price,
    set_symbol_meta,
    get_trail_state,
    set_trail_state,
    clear_trail_state,
    enrich_atr,  # used for sizing new symbols
    submit_market_order_with_fill,  # slippage logging
)

print("execution worker alive", flush=True)

# ============================================================
# CONFIG
# ============================================================
SLEEP_SECONDS = 300
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))  # HARD CAP (non-vol)
REBALANCE_COOLDOWN_SEC = int(os.getenv("REBALANCE_COOLDOWN_SEC", "1800"))  # 30m default

MIN_TRADE_DOLLARS = 500
NO_TRADE_BAND = 0.02
MIN_HOLD_DAYS = 3
EXIT_CONFIRM_RUNS = 3
REENTRY_COOLDOWN_SEC = 3600
EXEC_POS_STATE_KEY = "exec_positions"

STOP_ATR_MULT = 3.0

VOL_SLEEVE_SYMBOLS = {"SPY", "TLT"}
VOL_SLEEVE_STRATEGY = "vol_sleeve"

# Strategy priority for admission (lower is higher priority)
STRATEGY_PRIORITY = {
    "core_tech": 0,
    "continuation": 1,
    "momentum": 2,
    "mean_reversion": 3,
}

# Slippage log settings
TRADE_LOG_KEY = "trade_log"
TRADE_LOG_MAX = int(os.getenv("TRADE_LOG_MAX", "800"))

# ============================================================
# MARKET HOURS CHECK (US equities)
# ============================================================
def market_open() -> bool:
    """
    Returns True if US equity market is open (approximate).
    Used only to block invalid orders outside RTH.
    """
    now = datetime.now(timezone.utc)

    # Convert UTC → US Eastern (simple offset; good enough for execution guard)
    et = now - timedelta(hours=5)

    # Weekdays only
    if et.weekday() >= 5:
        return False

    # Regular trading hours
    return dt_time(9, 30) <= et.time() < dt_time(16, 0)


# ============================================================
# HELPERS
# ============================================================
def kill_switch_on() -> bool:
    return bool(STATE_LOG.get("governance", {}).get("kill_switch", False))

def vol_brake_on() -> bool:
    return bool((STATE_LOG.get("volatility_brake", {}) or {}).get("active", False))

def emergency_mode() -> bool:
    return kill_switch_on() or vol_brake_on()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def dollars_to_qty(symbol: str, dollars: float) -> int:
    if dollars <= 0:
        return 0
    price = get_last_trade_price(symbol)
    if price <= 0:
        return 0
    return int(math.floor(dollars / price))

def is_vol_sleeve_symbol(sym: str) -> bool:
    return sym in VOL_SLEEVE_SYMBOLS
def has_open_order(broker: Dict[str, Any], symbol: str) -> bool:
    """
    Prevent duplicate execution while an order for this symbol
    is still open / accepted / partially filled.
    """
    orders = broker.get("orders") or broker.get("open_orders") or []
    sym = symbol.upper()

    for o in orders:
        if not isinstance(o, dict):
            continue
        if str(o.get("symbol", "")).upper() != sym:
            continue
        status = str(o.get("status", "")).lower()
        if status in ("new", "accepted", "pending_new", "partially_filled"):
            return True
    return False

def uniq_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

def compute_exposure(alpha_targets: Dict[str, float], equity: float) -> Dict[str, float]:
    if equity <= 0:
        return {"gross": 0.0, "net": 0.0}
    gross = sum(abs(v) for v in alpha_targets.values()) / equity
    net = sum(v for v in alpha_targets.values()) / equity
    return {"gross": round(gross, 4), "net": round(net, 4)}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _append_trade_log(rec: Dict[str, Any]) -> None:
    cur = STATE_LOG.get(TRADE_LOG_KEY, [])
    if not isinstance(cur, list):
        cur = []
    cur.append(rec)
    if len(cur) > TRADE_LOG_MAX:
        cur = cur[-TRADE_LOG_MAX:]
    update_state(TRADE_LOG_KEY, cur)

# ============================================================
# EXECUTION STATE
# ============================================================
def _get_exec_state() -> Dict[str, Any]:
    return STATE_LOG.get(EXEC_POS_STATE_KEY, {}) or {}

def get_exec(symbol: str) -> Dict[str, Any]:
    return _get_exec_state().get(symbol, {}) or {}

def set_exec(symbol: str, patch: Dict[str, Any]) -> None:
    state = _get_exec_state()
    cur = state.get(symbol, {}) or {}
    cur.update(patch)
    state[symbol] = cur
    update_state(EXEC_POS_STATE_KEY, state)

def clear_exec(symbol: str) -> None:
    state = _get_exec_state()
    if symbol in state:
        del state[symbol]
        update_state(EXEC_POS_STATE_KEY, state)

def days_held(symbol: str) -> int:
    ed = get_exec(symbol).get("entry_date")
    if not ed:
        return 9999
    try:
        return (date.today() - date.fromisoformat(ed)).days
    except Exception:
        return 9999

def record_exit(symbol: str) -> None:
    set_exec(symbol, {"last_exit_ts": _now_iso(), "miss_count": 0})

def reentry_blocked(symbol: str) -> bool:
    ts = get_exec(symbol).get("last_exit_ts")
    if not ts:
        return False
    try:
        last = datetime.fromisoformat(ts)
        return (datetime.now(timezone.utc) - last).total_seconds() < REENTRY_COOLDOWN_SEC
    except Exception:
        return False

def recently_rebalanced(symbol: str) -> bool:
    """
    Prevent repeated delta-nudging (e.g., AAPL stacking) by enforcing
    a per-symbol cooldown after any executed trade.
    """
    ts = get_exec(symbol).get("last_rebalance_ts")
    if not ts:
        return False
    try:
        last = datetime.fromisoformat(ts)
        return (datetime.now(timezone.utc) - last).total_seconds() < REBALANCE_COOLDOWN_SEC
    except Exception:
        return False

def stamp_rebalanced(symbol: str) -> None:
    set_exec(symbol, {"last_rebalance_ts": _now_iso()})

def no_trade_band_ok(equity: float, cur: float, tgt: float) -> bool:
    if equity <= 0:
        return True
    return abs((tgt / equity) - (cur / equity)) >= NO_TRADE_BAND

# ============================================================
# HARD ENFORCEMENT: MAX POSITIONS (NON-VOL)
# ============================================================
def enforce_max_positions(alpha_targets, alpha_strategy, current_positions, max_positions):
    current_syms = [s for s in current_positions.keys() if not is_vol_sleeve_symbol(s)]
    current_set = set(current_syms)

    desired_syms = uniq_preserve(list(alpha_targets.keys()))

    def priority(sym: str):
        return (
            0 if sym in current_set else 1,  # keep existing first
            STRATEGY_PRIORITY.get(alpha_strategy.get(sym, ""), 99),
        )

    desired_syms.sort(key=priority)
    kept = desired_syms[:max_positions]

    dropped = set(alpha_targets.keys()) - set(kept)
    if dropped:
        print(f"EXEC → symbol cap enforced, dropped={sorted(dropped)}", flush=True)

    return {s: alpha_targets[s] for s in kept}

# ============================================================
# ATR-BASED SIZING (risk parity within sleeve)
# ============================================================
def allocate_sleeve_dollars_by_inv_atr(
    symbols: List[str],
    sleeve_dollars: float,
    atr_map: Dict[str, float],
) -> Dict[str, float]:
    """
    Distributes sleeve_dollars across symbols in proportion to 1/ATR.
    Falls back to equal-weight if ATR missing.
    """
    symbols = [str(s).upper() for s in symbols if s]
    if not symbols or sleeve_dollars <= 0:
        return {}

    inv = []
    for s in symbols:
        atr = atr_map.get(s)
        if atr is None or atr <= 0:
            inv.append(None)
        else:
            inv.append(1.0 / float(atr))

    if all(v is None for v in inv):
        per = sleeve_dollars / len(symbols)
        return {s: per for s in symbols}

    present = [v for v in inv if v is not None]
    med = sorted(present)[len(present)//2] if present else 1.0
    inv2 = [(v if v is not None else med) for v in inv]
    total = sum(inv2) if sum(inv2) > 0 else 1.0

    out = {}
    for s, w in zip(symbols, inv2):
        out[s] = sleeve_dollars * (w / total)
    return out

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    try:
        broker = read_only_check()
        equity = float(broker.get("equity", 0.0) or 0.0)
        cash = float(broker.get("cash", 0.0) or 0.0)
        positions = broker.get("positions", {}) or {}

        allocator = STATE_LOG.get("allocator", {}) or {}
        targets = allocator.get("targets", []) or []
        regime = allocator.get("regime", "Unknown")

        # Build a live ATR map from current positions (already enriched by broker)
        live_atr_map: Dict[str, float] = {}
        for sym, pos in positions.items():
            if isinstance(pos, dict):
                atr = pos.get("atr")
                if atr is not None:
                    try:
                        live_atr_map[str(sym).upper()] = float(atr)
                    except Exception:
                        pass

        alpha_targets: Dict[str, float] = {}
        alpha_strategy: Dict[str, str] = {}
        vol_targets: Dict[str, float] = {}

        desired_alpha_syms: List[str] = []

        # 1) Collect sleeve intentions and desired symbols
        sleeve_intents: List[Dict[str, Any]] = []
        for t in targets:
            syms = t.get("symbols", []) or []
            strat = (t.get("strategy") or "").strip()
            sleeve_dollars = safe_float(t.get("target_exposure", 0.0)) * equity
            if not syms or sleeve_dollars <= 0:
                continue

            # Separate vol sleeve
            if strat == VOL_SLEEVE_STRATEGY or any(is_vol_sleeve_symbol(s) for s in syms):
                per = sleeve_dollars / len(syms)
                for s in syms:
                    vol_targets[str(s).upper()] = per
                continue

            syms_u = [str(s).upper() for s in syms if s]
            desired_alpha_syms.extend(syms_u)
            sleeve_intents.append({"strategy": strat, "symbols": syms_u, "dollars": sleeve_dollars})

        desired_alpha_syms = uniq_preserve(desired_alpha_syms)

        # 2) Enrich ATR for desired symbols missing ATR
        missing = [s for s in desired_alpha_syms if s not in live_atr_map]
        if missing:
            try:
                extra = enrich_atr(missing, lookback_days=90) or {}
                for s, v in extra.items():
                    if v is None:
                        continue
                    try:
                        live_atr_map[str(s).upper()] = float(v)
                    except Exception:
                        pass
            except Exception as e:
                print(f"EXEC → ATR enrich failed: {e}", flush=True)

        # 3) Allocate sleeve dollars by inverse ATR
        for intent in sleeve_intents:
            strat = intent["strategy"]
            syms = intent["symbols"]
            sleeve_dollars = float(intent["dollars"])

            alloc_map = allocate_sleeve_dollars_by_inv_atr(syms, sleeve_dollars, live_atr_map)
            for s, dollars in alloc_map.items():
                alpha_targets[s] = alpha_targets.get(s, 0.0) + float(dollars)
                alpha_strategy[s] = strat

        # 4) HARD CAP (non-vol): trim down to MAX_POSITIONS
        alpha_targets = enforce_max_positions(alpha_targets, alpha_strategy, positions, MAX_POSITIONS)

        # ----------------------------------------------------
        # EXITS
        # ----------------------------------------------------
        for sym, pos in list(positions.items()):
            if is_vol_sleeve_symbol(sym):
                continue
            if sym in alpha_targets:
                set_exec(sym, {"miss_count": 0})
                continue

            if emergency_mode():
                close_position(sym)
                clear_trail_state(sym)
                record_exit(sym)
                clear_exec(sym)
                continue

            if days_held(sym) < MIN_HOLD_DAYS:
                continue

            miss = int(get_exec(sym).get("miss_count", 0)) + 1
            if miss < EXIT_CONFIRM_RUNS:
                set_exec(sym, {"miss_count": miss})
                continue

            close_position(sym)
            clear_trail_state(sym)
            record_exit(sym)
            clear_exec(sym)

        # ----------------------------------------------------
        # ENTRIES / REBALANCE (slippage logging + cooldown + qty guards)
        # ----------------------------------------------------
        for sym, tgt in alpha_targets.items():
            # HARD GUARD — prevents order ballooning
            if has_open_order(broker, sym):
                continue

            if reentry_blocked(sym):
                continue

            # Prevent repeated delta nudging (AAPL stacking)
            if recently_rebalanced(sym):
                continue

            pos = positions.get(sym)
            cur = float(pos.get("market_value", 0.0)) if isinstance(pos, dict) else 0.0
            delta = tgt - cur

            if not no_trade_band_ok(equity, cur, tgt):
                continue
            if abs(delta) < MIN_TRADE_DOLLARS:
                continue

            side = "buy" if delta > 0 else "sell"
            if side == "buy" and kill_switch_on():
                continue
            # HARD BLOCK: do not buy when market is closed
            if side == "buy" and not market_open():
                print(f"EXEC → blocked buy {sym}: market closed", flush=True)
                continue

            # --------- QTY COMPUTED FIRST (CRITICAL FIX) ---------
            qty = dollars_to_qty(sym, abs(delta))
            if qty <= 0:
                continue

            # HARD BLOCK: never submit sell orders without available shares
            if side == "sell":
                if not isinstance(pos, dict):
                    continue

                have_qty = safe_float(pos.get("available_qty", 0.0), 0.0)
                if have_qty <= 0:
                    print(f"EXEC → blocked sell {sym}: available_qty=0", flush=True)
                    continue

                qty = min(qty, int(have_qty))
                if qty <= 0:
                    continue

            try:
                decision_px = float(get_last_trade_price(sym))
            except Exception:
                decision_px = None

            fill_info = submit_market_order_with_fill(sym, qty, side)
            filled_avg = fill_info.get("filled_avg_price")
            status = fill_info.get("status")

            # Stamp cooldown ONLY when an order is accepted/filled
            if status in ("accepted", "filled", "partially_filled"):
                stamp_rebalanced(sym)

            # Log trade + slippage if benchmark + fill
            if decision_px is not None and filled_avg is not None and filled_avg > 0:
                if side == "buy":
                    slip = (filled_avg - decision_px) / decision_px
                else:
                    slip = (decision_px - filled_avg) / decision_px

                _append_trade_log({
                    "ts": _now_iso(),
                    "symbol": sym,
                    "side": side,
                    "qty": float(qty),
                    "strategy": alpha_strategy.get(sym),
                    "decision_px": float(decision_px),
                    "fill_px": float(filled_avg),
                    "slippage_frac": float(slip),
                    "slippage_bps": float(slip) * 10000.0,
                    "order_status": status,
                    "order_id": fill_info.get("id"),
                    "regime": regime,
                })

            # Update meta/state on buys
            if side == "buy":
                set_symbol_meta(sym, {
                    "strategy": alpha_strategy.get(sym, "momentum"),
                    "layers": 1,
                    "last_entry_price": decision_px if decision_px is not None else filled_avg,
                })
                if cur <= 0:
                    set_exec(sym, {"entry_date": date.today().isoformat(), "miss_count": 0})

        # ----------------------------------------------------
        # TRAILING STOPS
        # ----------------------------------------------------
        for sym, pos in positions.items():
            if is_vol_sleeve_symbol(sym) or not isinstance(pos, dict):
                continue
            qty = float(pos.get("qty", 0.0) or 0.0)
            price = pos.get("current_price")
            atr = pos.get("atr")
            if qty <= 0 or price is None or atr is None:
                continue

            ts = get_trail_state(sym)
            hwm = price if ts.get("high_water") is None else max(ts["high_water"], price)
            stop = hwm - STOP_ATR_MULT * atr
            set_trail_state(sym, {"high_water": hwm, "last_stop": stop})

            if price <= stop:
                close_position(sym)
                clear_trail_state(sym)
                record_exit(sym)
                clear_exec(sym)

        exposure = compute_exposure(alpha_targets, equity)

        update_state("execution", {
            "status": "ok",
            "equity": equity,
            "cash": cash,
            "positions": len([s for s in positions if not is_vol_sleeve_symbol(s)]),
            "gross_exposure": exposure["gross"],
            "net_exposure": exposure["net"],
            "max_positions": MAX_POSITIONS,
            "cap_pressure": max(0, len(alpha_targets) - MAX_POSITIONS),
            "regime": regime,
            "ts": _now_iso(),
        })

    except Exception as e:
        update_state("execution", {
            "status": "error",
            "error": str(e),
            "timestamp": _now_iso(),
        })
        print("EXECUTION ERROR:", e, flush=True)

    _time.sleep(SLEEP_SECONDS)
