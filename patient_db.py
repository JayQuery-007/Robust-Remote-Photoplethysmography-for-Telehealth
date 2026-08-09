"""patient_db.py — Patient authentication and medical history for EquiPhys.

SQLite-based, zero external cloud dependency. One file (patients.db) sits
next to streamlit_app.py. Passwords are hashed with bcrypt (never stored
in plain text).

Tables
------
users           — login credentials (user_id, username, role, password_hash)
patients        — full profile (linked to user_id)
vitals_sessions — per-session vitals snapshot (linked to patient_id)

Roles: 'patient' | 'doctor'
Doctors can view any patient. Patients see only their own data.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Database location ─────────────────────────────────────────────────────────
_DB_PATH = Path(__file__).resolve().parent / "patients.db"


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _hash_password(password: str) -> str:
    """SHA-256 PBKDF2 hash. No external deps (bcrypt not always available)."""
    salt = os.urandom(32)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return salt.hex() + ":" + key.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        key  = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
        return hmac.compare_digest(key, new_key)
    except Exception:
        return False


# ── Schema initialisation ─────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = _get_conn()
    cur  = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
        role          TEXT    NOT NULL DEFAULT 'patient',  -- 'patient' | 'doctor'
        password_hash TEXT    NOT NULL,
        created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS patients (
        patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id         INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        full_name       TEXT    NOT NULL,
        date_of_birth   TEXT,          -- YYYY-MM-DD
        sex             TEXT,          -- 'Male' | 'Female' | 'Other'
        blood_group     TEXT,
        fitzpatrick     INTEGER,       -- 1-6 (Fitzpatrick skin type)
        height_cm       REAL,
        weight_kg       REAL,
        conditions      TEXT,          -- JSON list e.g. ["Hypertension","Diabetes"]
        medications     TEXT,          -- JSON list e.g. ["Metformin 500mg"]
        allergies       TEXT,          -- JSON list
        notes           TEXT,          -- free-text clinical notes
        updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS vitals_sessions (
        session_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id      INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
        recorded_at     TEXT    NOT NULL DEFAULT (datetime('now')),
        duration_s      REAL,          -- session duration in seconds
        hr_bpm          REAL,          -- mean HR (BPM)
        hr_min          REAL,
        hr_max          REAL,
        spo2_pct        REAL,          -- SpO2 %
        rr_bpm          REAL,          -- respiratory rate (br/min)
        sbp_mmhg        REAL,          -- systolic BP (if calibrated)
        dbp_mmhg        REAL,          -- diastolic BP (if calibrated)
        sqi             REAL,          -- mean signal quality index (0-100)
        snr_db          REAL,          -- mean SNR (dB)
        fps             REAL,          -- capture frame rate
        method          TEXT,          -- 'POS' | 'CHROM' | 'Fusion'
        notes           TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_vitals_patient ON vitals_sessions(patient_id);
    CREATE INDEX IF NOT EXISTS idx_vitals_time    ON vitals_sessions(recorded_at);
    """)
    conn.commit()
    conn.close()


# ── User / Auth ───────────────────────────────────────────────────────────────

def register_user(username: str, password: str,
                  role: str = "patient") -> Tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if role not in ("patient", "doctor"):
        return False, "Invalid role."

    pw_hash = _hash_password(password)
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO users (username, role, password_hash) VALUES (?, ?, ?)",
            (username.strip(), role, pw_hash),
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' is already taken."
    except Exception as e:
        return False, f"Database error: {e}"


def login_user(username: str, password: str) -> Optional[Dict]:
    """Verify credentials. Returns user dict or None."""
    try:
        conn = _get_conn()
        row  = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
        conn.close()
        if row and _verify_password(password, row["password_hash"]):
            return dict(row)
        return None
    except Exception:
        return None


def get_all_patients_for_doctor() -> List[Dict]:
    """Doctors only — list all patients with basic info."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT p.patient_id, p.full_name, p.date_of_birth, p.sex,
               u.username, p.updated_at,
               COUNT(vs.session_id) AS session_count
        FROM patients p
        JOIN users u ON u.user_id = p.user_id
        LEFT JOIN vitals_sessions vs ON vs.patient_id = p.patient_id
        GROUP BY p.patient_id
        ORDER BY p.full_name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Patient Profile ───────────────────────────────────────────────────────────

def get_patient_by_user(user_id: int) -> Optional[Dict]:
    conn = _get_conn()
    row  = conn.execute(
        "SELECT * FROM patients WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        for f in ("conditions", "medications", "allergies"):
            if d.get(f):
                try:
                    d[f] = json.loads(d[f])
                except Exception:
                    d[f] = [d[f]]
            else:
                d[f] = []
        return d
    return None


def get_patient_by_id(patient_id: int) -> Optional[Dict]:
    conn = _get_conn()
    row  = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        for f in ("conditions", "medications", "allergies"):
            if d.get(f):
                try:
                    d[f] = json.loads(d[f])
                except Exception:
                    d[f] = [d[f]]
            else:
                d[f] = []
        return d
    return None


def upsert_patient(user_id: int, data: Dict) -> Tuple[bool, str]:
    """Create or update patient profile."""
    for f in ("conditions", "medications", "allergies"):
        if isinstance(data.get(f), list):
            data[f] = json.dumps(data[f])

    try:
        conn   = _get_conn()
        exists = conn.execute(
            "SELECT patient_id FROM patients WHERE user_id = ?", (user_id,)
        ).fetchone()

        if exists:
            conn.execute("""
                UPDATE patients SET
                    full_name=?, date_of_birth=?, sex=?, blood_group=?,
                    fitzpatrick=?, height_cm=?, weight_kg=?,
                    conditions=?, medications=?, allergies=?, notes=?,
                    updated_at=datetime('now')
                WHERE user_id=?
            """, (
                data.get("full_name",""), data.get("date_of_birth"),
                data.get("sex"), data.get("blood_group"),
                data.get("fitzpatrick"), data.get("height_cm"),
                data.get("weight_kg"), data.get("conditions"),
                data.get("medications"), data.get("allergies"),
                data.get("notes"), user_id,
            ))
        else:
            conn.execute("""
                INSERT INTO patients
                    (user_id, full_name, date_of_birth, sex, blood_group,
                     fitzpatrick, height_cm, weight_kg,
                     conditions, medications, allergies, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                user_id, data.get("full_name",""), data.get("date_of_birth"),
                data.get("sex"), data.get("blood_group"),
                data.get("fitzpatrick"), data.get("height_cm"),
                data.get("weight_kg"), data.get("conditions"),
                data.get("medications"), data.get("allergies"),
                data.get("notes"),
            ))
        conn.commit()
        conn.close()
        return True, "Profile saved."
    except Exception as e:
        return False, f"Error saving profile: {e}"


# ── Vitals Sessions ───────────────────────────────────────────────────────────

def save_session(patient_id: int, data: Dict) -> Tuple[bool, str]:
    """Save one vitals measurement session."""
    try:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO vitals_sessions
                (patient_id, duration_s, hr_bpm, hr_min, hr_max,
                 spo2_pct, rr_bpm, sbp_mmhg, dbp_mmhg,
                 sqi, snr_db, fps, method, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            patient_id,
            data.get("duration_s"),
            data.get("hr_bpm"),
            data.get("hr_min"),
            data.get("hr_max"),
            data.get("spo2_pct"),
            data.get("rr_bpm"),
            data.get("sbp_mmhg"),
            data.get("dbp_mmhg"),
            data.get("sqi"),
            data.get("snr_db"),
            data.get("fps"),
            data.get("method", "Fusion"),
            data.get("notes"),
        ))
        conn.commit()
        conn.close()
        return True, "Session saved."
    except Exception as e:
        return False, f"Error saving session: {e}"


def get_sessions(patient_id: int, limit: int = 50) -> List[Dict]:
    """Get last N sessions for a patient, newest first."""
    conn  = _get_conn()
    rows  = conn.execute("""
        SELECT * FROM vitals_sessions
        WHERE patient_id = ?
        ORDER BY recorded_at DESC
        LIMIT ?
    """, (patient_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_stats(patient_id: int) -> Dict:
    """Aggregate statistics across all sessions for a patient."""
    conn = _get_conn()
    row  = conn.execute("""
        SELECT
            COUNT(*)            AS total_sessions,
            ROUND(AVG(hr_bpm),1)  AS avg_hr,
            ROUND(MIN(hr_bpm),1)  AS min_hr,
            ROUND(MAX(hr_bpm),1)  AS max_hr,
            ROUND(AVG(spo2_pct),1) AS avg_spo2,
            ROUND(AVG(rr_bpm),1)  AS avg_rr,
            MIN(recorded_at)    AS first_session,
            MAX(recorded_at)    AS last_session
        FROM vitals_sessions
        WHERE patient_id = ?
    """, (patient_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}
