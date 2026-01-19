import os
import psycopg2
import json
from datetime import datetime

# ============================================================
# DATABASE URL (Render-safe)
# ============================================================
DB_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("MODEL_STATE_DB_URL")
)

print("MODEL_STATE_LOGGER LOADED FROM:", __file__, flush=True)
print("MODEL_STATE_LOGGER DB_URL:", (DB_URL[:40] + "...") if DB_URL else "NONE", flush=True)

if not DB_URL:
    raise RuntimeError("No DATABASE_URL or MODEL_STATE_DB_URL set")

# ============================================================
# INIT
# ============================================================
def init_model_state_db():
    try:
        conn = psycopg2.connect(DB_URL, sslmode="require")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_state_log (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT NOT NULL,
                regime TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_posture TEXT,
                target_gross REAL,
                sleeves TEXT,
                leaders TEXT,
                state_changed INTEGER DEFAULT 0,
                human_reviewed INTEGER DEFAULT 0
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Postgres model_state_log table ensured", flush=True)
    except Exception as e:
        print("MODEL_STATE_LOGGER INIT ERROR:", e, flush=True)

# ============================================================
# LOGGING
# ============================================================
def log_model_state(
    model_version: str,
    regime: str,
    confidence: float,
    risk_posture: str | None,
    target_gross: float | None,
    sleeves,
    leaders,
    state_changed: bool = False,
    human_reviewed: bool = False,
):
    print("MODEL_STATE_LOGGER → log_model_state() CALLED", flush=True)

    try:
        # Ensure table exists (safe + idempotent)
        init_model_state_db()

        conn = psycopg2.connect(DB_URL, sslmode="require")
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO model_state_log (
                model_version,
                regime,
                confidence,
                risk_posture,
                target_gross,
                sleeves,
                leaders,
                state_changed,
                human_reviewed
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            model_version,
            regime,
            float(confidence),
            risk_posture,
            target_gross,
            json.dumps(sleeves) if sleeves is not None else None,
            json.dumps(leaders) if leaders is not None else None,
            int(state_changed),
            int(human_reviewed),
        ))

        conn.commit()
        cur.close()
        conn.close()

        print(f"MODEL_STATE_LOGGER → Logged {regime} (confidence {confidence})", flush=True)

    except Exception as e:
        # IMPORTANT: allocator must NOT crash if Postgres blips
        print("MODEL_STATE_LOGGER WRITE ERROR:", e, flush=True)
