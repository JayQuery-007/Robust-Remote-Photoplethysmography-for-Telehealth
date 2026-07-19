"""streamlit_vitals_integration.py — drop-in UI wiring for facial_vitals.

Replaces the fabricated BP/Hb heuristic and the fake-clamped SpO2 with
provenance-aware cards driven by facial_vitals.estimate_all_vitals().

How to use in streamlit_app.py
------------------------------
1)  from facial_vitals import estimate_all_vitals, VitalsCalibration
    from streamlit_vitals_integration import sidebar_calibration_controls, render_vitals_panel

2)  In the sidebar block (near the other checkboxes):
        calib = sidebar_calibration_controls(st, st.session_state)

3)  Replace the per-frame HR/SpO2/RR estimation + the "Display metrics" block
    (streamlit_app.py lines ~578-784) with, once per inference tick:

        vitals = estimate_all_vitals(
            frame_buffer=list(trad_roi_buffer),
            timestamps=np.asarray(list(trad_ts_buffer), dtype=float),
            fps=fs_effective,
            calib=calib,
        )
        render_vitals_panel(st, vitals, slots)          # slots: dict of st.empty()

    You can keep the existing motion gate, EMA smoothing and HR-lock logic by
    feeding vitals["heart_rate"].value into them instead of the old `hr_raw`.

The temporal smoothing / lock / harmonic-correction machinery you already have
is good and orthogonal to this change — this file only fixes *what the numbers
mean* and *how honestly they are shown*.
"""
from __future__ import annotations

from typing import Dict

from facial_vitals import (
    PersonalizedBPEstimator,
    Spo2Calibration,
    VitalSign,
    VitalsCalibration,
)

SRC_LABEL = {
    "rppg_spectral": "measured",
    "rppg_envelope": "measured",
    "rppg_ratio_calibrated": "calibrated",
    "rppg_ratio_uncalibrated": "provisional",
    "rppg_morphology_personalized": "trend vs baseline",
    "rppg_morphology_model": "model",
    "unavailable": "unavailable",
}


def sidebar_calibration_controls(st, session_state) -> VitalsCalibration:
    """Render calibration inputs and return a VitalsCalibration.

    Persists the BP estimator across reruns via session_state so a captured cuff
    baseline / trained model survives Streamlit's re-execution model.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Calibration")

    if "bp_estimator" not in session_state:
        session_state.bp_estimator = PersonalizedBPEstimator()
    bp = session_state.bp_estimator

    # ---- SpO2 device calibration ----------------------------------------- #
    with st.sidebar.expander("SpO₂ device calibration", expanded=False):
        st.caption("Enter one or more reference-oximeter readings taken on THIS "
                   "camera + lighting. Without this, SpO₂ is provisional only.")
        ref = st.text_input("ref% , ratioR  (one pair per line)",
                            value="", key="spo2_ref",
                            help="e.g.\n98, 0.55\n95, 0.80")
        spo2_cal = None
        if ref.strip():
            pcts, ratios = [], []
            for line in ref.splitlines():
                try:
                    p, r = line.replace(";", ",").split(",")
                    pcts.append(float(p)); ratios.append(float(r))
                except Exception:
                    continue
            if pcts:
                spo2_cal = Spo2Calibration.fit(ratios=ratios, spo2_ref=pcts)
                st.success(f"SpO₂ calibrated ({spo2_cal.note}).")

    allow_prov = st.sidebar.checkbox("Show provisional (uncalibrated) SpO₂",
                                     value=True,
                                     help="Uncalibrated SpO₂ is a relative trend, "
                                          "not a validated percentage.")

    # ---- BP cuff anchor (auto-applied inside the live loop) -------------- #
    with st.sidebar.expander("Blood pressure (cuff anchor)", expanded=True):
        st.caption("Absolute BP cannot come from a face alone. Enter ONE cuff "
                   "reading; BP is then reported as a TREND anchored to it. "
                   "Set the values, tick the box, THEN start live inference.")
        c1, c2 = st.columns(2)
        sbp0 = c1.number_input("Cuff SBP", 70, 220, 120, key="bp_sbp0")
        dbp0 = c2.number_input("Cuff DBP", 40, 140, 80, key="bp_dbp0")
        anchor_on = st.checkbox("Anchor BP to this cuff reading (live)",
                                value=False, key="bp_anchor_on")
        if bp.baseline is not None:
            st.success(f"Anchored at {bp.baseline.sbp:.0f}/{bp.baseline.dbp:.0f} mmHg "
                       "— card now shows trend.")
        elif anchor_on:
            st.info("Waiting for a stable pulse to anchor…")

    calib = VitalsCalibration(spo2=spo2_cal, bp=bp, allow_provisional_spo2=allow_prov)
    calib.bp_anchor = (float(sbp0), float(dbp0)) if anchor_on else None
    return calib


def _card(st, label: str, v: VitalSign, unit_override: str = "") -> None:
    tag = SRC_LABEL.get(v.source, v.source)
    unit = unit_override or v.unit
    if not v.available:
        val = "—"
        sub = f"unavailable · {v.note}" if v.note else "unavailable"
    else:
        val = f"{v.value:.0f}"
        bars = int(round(v.confidence * 5))
        sub = f"{tag} · confidence {'▮'*bars}{'▯'*(5-bars)}"
        if v.source in ("rppg_ratio_uncalibrated", "rppg_morphology_personalized"):
            sub += f" · {v.note}"
    st.markdown(
        f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;
                    padding:16px 12px;text-align:center;min-height:150px;
                    display:flex;flex-direction:column;justify-content:center;">
          <div style="color:#94a3b8;font-size:12px;font-weight:600;letter-spacing:.5px;">{label}</div>
          <div style="font-size:44px;font-weight:700;color:#f8fafc;line-height:1;
                      font-variant-numeric:tabular-nums;margin:6px 0;">{val}</div>
          <div style="color:#64748b;font-size:13px;font-weight:500;">{unit}</div>
          <div style="color:#475569;font-size:10.5px;margin-top:8px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_vitals_panel(st, vitals: Dict[str, VitalSign], slots: Dict[str, object],
                        session_state=None) -> None:
    """Render HR / SpO₂ / RR / BP cards with provenance + confidence.

    `slots` maps 'hr','spo2','rr','bp' to st.empty() placeholders.
    Pass session_state to stash morphology for BP-baseline capture.
    """
    if session_state is not None:
        sq = vitals.get("signal_quality")
        if sq is not None and sq.extra:
            session_state["last_morph"] = sq.extra.get("morphology")

    with slots["hr"]:
        _card(st, "HEART RATE", vitals["heart_rate"], "BPM")
    with slots["spo2"]:
        _card(st, "SpO₂", vitals["spo2"], "%")
    with slots["rr"]:
        _card(st, "RESP RATE", vitals["respiratory_rate"], "br/min")
    if "bp" in slots:
        sbp, dbp = vitals["systolic_bp"], vitals["diastolic_bp"]
        with slots["bp"]:
            if sbp.available and dbp.available:
                bp_view = VitalSign("blood_pressure", sbp.value, "mmHg",
                                    min(sbp.confidence, dbp.confidence), sbp.source,
                                    sbp.note)
                # show as "SBP/DBP"
                tag = SRC_LABEL.get(sbp.source, sbp.source)
                st.markdown(
                    f"""
                    <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;
                                padding:16px 12px;text-align:center;min-height:150px;
                                display:flex;flex-direction:column;justify-content:center;">
                      <div style="color:#94a3b8;font-size:12px;font-weight:600;">BLOOD PRESSURE</div>
                      <div style="font-size:40px;font-weight:700;color:#f8fafc;margin:6px 0;
                                  font-variant-numeric:tabular-nums;">{sbp.value:.0f}/{dbp.value:.0f}</div>
                      <div style="color:#64748b;font-size:13px;">mmHg</div>
                      <div style="color:#475569;font-size:10.5px;margin-top:8px;">{tag} · {sbp.note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                _card(st, "BLOOD PRESSURE", sbp, "mmHg")