"""patient_ui.py — Patient history UI for EquiPhys.

Pages:
  login_page()        — auth gate
  profile_page()      — view/edit patient profile
  history_page()      — vitals history + charts
  all_patients_page() — doctor view of all patients
  render_auth_sidebar() — sidebar nav + logout
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import streamlit as st

from patient_db import (
    get_all_patients_for_doctor,
    get_patient_by_id,
    get_patient_by_user,
    get_session_stats,
    get_sessions,
    login_user,
    register_user,
    upsert_patient,
)

FITZPATRICK_LABELS = {
    1: "Type I — Very fair, always burns",
    2: "Type II — Fair, usually burns",
    3: "Type III — Medium, sometimes burns",
    4: "Type IV — Olive, rarely burns",
    5: "Type V — Brown, very rarely burns",
    6: "Type VI — Dark brown / black, never burns",
}
BLOOD_GROUPS = ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"]
NAV_PAGES   = ["Live Inference", "Patient Profile", "Vitals History"]
NAV_PAGES_DOCTOR = ["Live Inference", "Patient Profile", "Vitals History", "All Patients"]

# ── Shared page header ────────────────────────────────────────────────────────

_PAGE_HEADER_CSS = """
<style>
.eq-page-header {
    background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
    border-left: 4px solid #38bdf8;
    border-radius: 6px;
    padding: 18px 24px 14px 24px;
    margin-bottom: 28px;
}
.eq-page-header .eq-portal-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.eq-page-header .eq-page-title {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0;
}
.eq-page-header .eq-page-sub {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
}
.eq-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 20px 0;
}
.eq-section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #64748b;
    text-transform: uppercase;
    margin: 20px 0 8px 0;
}
</style>
"""

def _page_header(title: str, subtitle: str = "") -> None:
    st.markdown(_PAGE_HEADER_CSS, unsafe_allow_html=True)
    sub_html = f'<div class="eq-page-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="eq-page-header">
            <div class="eq-portal-label">EquiPhys Telehealth Portal</div>
            <div class="eq-page-title">{title}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_auth_sidebar() -> None:
    user   = st.session_state.get("auth_user", {})
    role   = user.get("role", "patient").capitalize()
    uname  = user.get("username", "—")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:13px;color:#94a3b8;margin-bottom:2px;'>Signed in as</div>"
        f"<div style='font-size:15px;font-weight:600;color:#e2e8f0;'>{uname}</div>"
        f"<div style='font-size:12px;color:#38bdf8;margin-bottom:10px;'>{role}</div>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Sign Out", use_container_width=True):
        for k in ["auth_user", "auth_patient", "active_page", "doctor_selected_patient"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.sidebar.markdown("---")
    pages = NAV_PAGES_DOCTOR if user.get("role") == "doctor" else NAV_PAGES
    current = st.session_state.get("active_page", "Live Inference")
    if current not in pages:
        current = pages[0]
    choice = st.sidebar.radio(
        "Navigation", pages,
        index=pages.index(current),
    )
    st.session_state["active_page"] = choice


# ── Login page ────────────────────────────────────────────────────────────────

def login_page() -> None:
    st.markdown(_PAGE_HEADER_CSS, unsafe_allow_html=True)

    # Centred login card
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown(
            """
            <div style='text-align:center; padding:40px 0 28px 0;'>
                <div style='font-size:11px;font-weight:600;letter-spacing:0.12em;
                            color:#38bdf8;text-transform:uppercase;'>EquiPhys</div>
                <div style='font-size:26px;font-weight:700;color:#e2e8f0;
                            margin:8px 0 4px 0;'>Telehealth Portal</div>
                <div style='font-size:13px;color:#64748b;'>
                    Remote Photoplethysmography — Vital Signs Monitoring
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_in, tab_reg = st.tabs(["Sign In", "Create Account"])

        with tab_in:
            username = st.text_input("Username", key="login_user",
                                     placeholder="Enter username")
            password = st.text_input("Password", type="password",
                                     key="login_pw", placeholder="Enter password")
            if st.button("Sign In", use_container_width=True, type="primary"):
                if username and password:
                    user = login_user(username, password)
                    if user:
                        st.session_state["auth_user"]    = user
                        st.session_state["active_page"]  = "Live Inference"
                        patient = get_patient_by_user(user["user_id"])
                        if patient:
                            st.session_state["auth_patient"] = patient
                        st.success("Signed in successfully.")
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")
                else:
                    st.warning("Please enter both username and password.")

        with tab_reg:
            new_user = st.text_input("Choose a username", key="reg_user")
            new_pw   = st.text_input("Password (min 6 characters)",
                                     type="password", key="reg_pw")
            new_pw2  = st.text_input("Confirm password",
                                     type="password", key="reg_pw2")
            new_role = st.selectbox("Account type",
                                    ["patient", "doctor"], key="reg_role",
                                    format_func=lambda x: x.capitalize())
            if st.button("Create Account", use_container_width=True, type="primary"):
                if new_pw != new_pw2:
                    st.error("Passwords do not match.")
                elif not new_user or not new_pw:
                    st.warning("Username and password are required.")
                else:
                    ok, msg = register_user(new_user, new_pw, new_role)
                    if ok:
                        st.success(msg + "  Please sign in.")
                    else:
                        st.error(msg)

        st.markdown(
            "<div style='font-size:11px;color:#334155;text-align:center;"
            "padding:20px 0 0 0;'>For research and demonstration purposes only."
            "  Not a certified medical device.</div>",
            unsafe_allow_html=True,
        )


# ── Profile page ──────────────────────────────────────────────────────────────

def profile_page(read_only: bool = False,
                 patient_id: Optional[int] = None) -> None:
    user = st.session_state.get("auth_user", {})

    if patient_id:
        profile = get_patient_by_id(patient_id)
        title   = profile["full_name"] if profile else "Patient"
        _page_header("Patient Profile", title)
    else:
        profile = (st.session_state.get("auth_patient")
                   or get_patient_by_user(user.get("user_id", 0)))
        _page_header("Patient Profile",
                     "View and update your personal and medical information")

    def _s(v): return str(v).strip() if v is not None else ""

    # ── Read-only view (doctor) ───────────────────────────────────────────────
    if read_only:
        if not profile:
            st.info("No profile on file for this patient.")
            return

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="eq-section-label">Personal</div>',
                        unsafe_allow_html=True)
            st.metric("Full Name",     profile.get("full_name") or "—")
            st.metric("Date of Birth", profile.get("date_of_birth") or "—")
            st.metric("Sex",           profile.get("sex") or "—")
            st.metric("Blood Group",   profile.get("blood_group") or "—")
            fitz = profile.get("fitzpatrick")
            st.metric("Fitzpatrick Type",
                      FITZPATRICK_LABELS.get(fitz, "—") if fitz else "—")
        with c2:
            st.markdown('<div class="eq-section-label">Biometrics</div>',
                        unsafe_allow_html=True)
            h = profile.get("height_cm")
            w = profile.get("weight_kg")
            st.metric("Height", f"{h} cm" if h else "—")
            st.metric("Weight", f"{w} kg" if w else "—")
            bmi = round(w / (h/100)**2, 1) if h and w else None
            st.metric("BMI", str(bmi) if bmi else "—")

        st.markdown('<hr class="eq-divider">', unsafe_allow_html=True)
        c3, c4, c5 = st.columns(3)
        with c3:
            st.markdown('<div class="eq-section-label">Conditions</div>',
                        unsafe_allow_html=True)
            conds = profile.get("conditions") or []
            for c in conds:
                st.markdown(f"— {c}")
            if not conds:
                st.caption("None recorded")
        with c4:
            st.markdown('<div class="eq-section-label">Medications</div>',
                        unsafe_allow_html=True)
            meds = profile.get("medications") or []
            for m in meds:
                st.markdown(f"— {m}")
            if not meds:
                st.caption("None recorded")
        with c5:
            st.markdown('<div class="eq-section-label">Allergies</div>',
                        unsafe_allow_html=True)
            algs = profile.get("allergies") or []
            for a in algs:
                st.markdown(f"— {a}")
            if not algs:
                st.caption("None recorded")

        if profile.get("notes"):
            st.markdown('<div class="eq-section-label">Clinical Notes</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f"<div style='background:#0f172a;border:1px solid #1e293b;"
                f"border-radius:6px;padding:12px 16px;color:#cbd5e1;"
                f"font-size:14px;'>{profile['notes']}</div>",
                unsafe_allow_html=True,
            )
        return

    # ── Editable form ─────────────────────────────────────────────────────────
    with st.form("profile_form"):
        st.markdown('<div class="eq-section-label">Personal Information</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("Full Name *",
                value=_s(profile.get("full_name")) if profile else "")
            dob = st.text_input("Date of Birth (YYYY-MM-DD)",
                value=_s(profile.get("date_of_birth")) if profile else "",
                placeholder="e.g. 1990-06-15")
            _sex_opts = ["", "Male", "Female", "Other", "Prefer not to say"]
            _sex_val  = _s(profile.get("sex")) if profile else ""
            sex = st.selectbox("Sex", _sex_opts,
                index=_sex_opts.index(_sex_val) if _sex_val in _sex_opts else 0)
            _bg_val = _s(profile.get("blood_group")) if profile else ""
            blood_group = st.selectbox("Blood Group", BLOOD_GROUPS,
                index=BLOOD_GROUPS.index(_bg_val) if _bg_val in BLOOD_GROUPS else 0)

        with c2:
            _h = float(profile.get("height_cm") or 0) if profile else 0.0
            _w = float(profile.get("weight_kg") or 0) if profile else 0.0
            height_cm = st.number_input("Height (cm)", 0.0, 300.0,
                                        value=_h, step=0.5)
            weight_kg = st.number_input("Weight (kg)", 0.0, 500.0,
                                        value=_w, step=0.5)
            fitz_options = [""] + [f"{k} — {v.split(' — ')[1]}"
                                   for k, v in FITZPATRICK_LABELS.items()]
            fitz_val = int(profile.get("fitzpatrick") or 0) if profile else 0
            fitz_idx = fitz_val if 1 <= fitz_val <= 6 else 0
            fitzpatrick = st.selectbox("Fitzpatrick Skin Type",
                                       fitz_options, index=fitz_idx)

        st.markdown('<div class="eq-section-label">Medical History</div>',
                    unsafe_allow_html=True)
        st.caption("Enter each item on a separate line.")
        c3, c4, c5 = st.columns(3)
        with c3:
            conds_txt = st.text_area("Medical Conditions",
                value="\n".join(profile.get("conditions") or []) if profile else "",
                height=130, placeholder="e.g.\nHypertension\nType 2 Diabetes")
        with c4:
            meds_txt = st.text_area("Current Medications",
                value="\n".join(profile.get("medications") or []) if profile else "",
                height=130, placeholder="e.g.\nMetformin 500mg\nAmlodipine 5mg")
        with c5:
            algs_txt = st.text_area("Allergies",
                value="\n".join(profile.get("allergies") or []) if profile else "",
                height=130, placeholder="e.g.\nPenicillin\nLatex")

        notes_txt = st.text_area("Clinical Notes",
            value=_s(profile.get("notes")) if profile else "",
            height=80,
            placeholder="Any additional clinical information...")

        submitted = st.form_submit_button("Save Profile",
                                          type="primary",
                                          use_container_width=True)

    if submitted:
        def _lines(t): return [l.strip() for l in (t or "").splitlines() if l.strip()]

        fitz_int = None
        if fitzpatrick and _s(fitzpatrick):
            try:
                fitz_int = int(_s(fitzpatrick).split(" — ")[0])
            except Exception:
                fitz_int = None

        data = {
            "full_name":     _s(full_name),
            "date_of_birth": _s(dob) or None,
            "sex":           sex or None,
            "blood_group":   blood_group or None,
            "fitzpatrick":   fitz_int,
            "height_cm":     height_cm if (height_cm or 0) > 0 else None,
            "weight_kg":     weight_kg if (weight_kg or 0) > 0 else None,
            "conditions":    _lines(conds_txt),
            "medications":   _lines(meds_txt),
            "allergies":     _lines(algs_txt),
            "notes":         _s(notes_txt) or None,
        }

        if not data["full_name"]:
            st.error("Full Name is required.")
        else:
            ok, msg = upsert_patient(user["user_id"], data)
            if ok:
                st.success("Profile saved.")
                st.session_state["auth_patient"] = get_patient_by_user(user["user_id"])
            else:
                st.error(msg)


# ── History page ──────────────────────────────────────────────────────────────

def history_page(patient_id: Optional[int] = None) -> None:
    import pandas as pd
    import altair as alt

    patient = (get_patient_by_id(patient_id) if patient_id
               else st.session_state.get("auth_patient"))

    if not patient:
        _page_header("Vitals History")
        st.warning("No patient profile found. Please complete your profile first.")
        if st.button("Go to Profile"):
            st.session_state["active_page"] = "Patient Profile"
            st.rerun()
        return

    pid   = patient["patient_id"]
    name  = patient.get("full_name", "Patient")
    stats = get_session_stats(pid)
    sessions = get_sessions(pid, limit=100)

    _page_header("Vitals History", name)

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sessions",  stats.get("total_sessions", 0))
    c2.metric("Avg HR",    f"{stats.get('avg_hr','—')} BPM"
              if stats.get("avg_hr") else "—")
    c3.metric("HR Range",  f"{stats.get('min_hr','—')} – {stats.get('max_hr','—')} BPM"
              if stats.get("min_hr") else "—")
    c4.metric("Avg SpO2",  f"{stats.get('avg_spo2','—')} %"
              if stats.get("avg_spo2") else "—")
    c5.metric("Avg RR",    f"{stats.get('avg_rr','—')} br/min"
              if stats.get("avg_rr") else "—")

    if not sessions:
        st.markdown('<hr class="eq-divider">', unsafe_allow_html=True)
        st.info("No sessions recorded yet. Start a live inference session to begin tracking.")
        return

    df = pd.DataFrame(sessions)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])

    _AXIS = lambda title: alt.Axis(
        title=title, labelColor="#94a3b8", titleColor="#94a3b8",
        gridColor="#1e293b", domainColor="#334155", tickColor="#334155"
    )
    _VIEW = {"strokeWidth": 0, "fill": "transparent"}
    _PROPS = {"background": "transparent",
              "padding": {"top": 6, "bottom": 6, "left": 4, "right": 4}}

    # ── HR trend ──────────────────────────────────────────────────────────────
    st.markdown('<div class="eq-section-label">Heart Rate</div>',
                unsafe_allow_html=True)
    hr_df = df[df["hr_bpm"].notna()]
    if not hr_df.empty:
        ch = (
            alt.Chart(hr_df)
            .mark_line(color="#38bdf8", strokeWidth=2,
                       point=alt.OverlayMarkDef(color="#38bdf8", size=50))
            .encode(
                x=alt.X("recorded_at:T", axis=_AXIS("Date / Time")),
                y=alt.Y("hr_bpm:Q", scale=alt.Scale(zero=False),
                        axis=_AXIS("HR (BPM)")),
                tooltip=[
                    alt.Tooltip("recorded_at:T", title="Time"),
                    alt.Tooltip("hr_bpm:Q", title="HR (BPM)", format=".0f"),
                    alt.Tooltip("spo2_pct:Q", title="SpO2 (%)", format=".1f"),
                    alt.Tooltip("method:N", title="Method"),
                ],
            )
            .properties(height=240, **_PROPS)
            .configure_view(**_VIEW)
        )
        st.altair_chart(ch, use_container_width=True)

    # ── SpO2 + RR side by side ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="eq-section-label">SpO2</div>',
                    unsafe_allow_html=True)
        spo2_df = df[df["spo2_pct"].notna()]
        if not spo2_df.empty:
            ch = (alt.Chart(spo2_df)
                  .mark_line(color="#34d399", strokeWidth=2)
                  .encode(
                      x=alt.X("recorded_at:T", axis=_AXIS("")),
                      y=alt.Y("spo2_pct:Q",
                              scale=alt.Scale(domain=[85, 101]),
                              axis=_AXIS("SpO2 (%)")),
                      tooltip=[alt.Tooltip("recorded_at:T", title="Time"),
                               alt.Tooltip("spo2_pct:Q", title="SpO2 (%)", format=".1f")],
                  )
                  .properties(height=200, **_PROPS)
                  .configure_view(**_VIEW))
            st.altair_chart(ch, use_container_width=True)
        else:
            st.caption("No SpO2 data recorded.")

    with col_b:
        st.markdown('<div class="eq-section-label">Respiratory Rate</div>',
                    unsafe_allow_html=True)
        rr_df = df[df["rr_bpm"].notna()]
        if not rr_df.empty:
            ch = (alt.Chart(rr_df)
                  .mark_line(color="#f59e0b", strokeWidth=2)
                  .encode(
                      x=alt.X("recorded_at:T", axis=_AXIS("")),
                      y=alt.Y("rr_bpm:Q",
                              scale=alt.Scale(zero=False),
                              axis=_AXIS("RR (br/min)")),
                      tooltip=[alt.Tooltip("recorded_at:T", title="Time"),
                               alt.Tooltip("rr_bpm:Q", title="RR (br/min)", format=".1f")],
                  )
                  .properties(height=200, **_PROPS)
                  .configure_view(**_VIEW))
            st.altair_chart(ch, use_container_width=True)
        else:
            st.caption("No respiratory rate data recorded.")

    # ── Session table ─────────────────────────────────────────────────────────
    st.markdown('<div class="eq-section-label">Session Log</div>',
                unsafe_allow_html=True)
    col_map = {
        "recorded_at": "Recorded At",
        "hr_bpm":      "HR (BPM)",
        "spo2_pct":    "SpO2 (%)",
        "rr_bpm":      "RR (br/min)",
        "sbp_mmhg":    "SBP (mmHg)",
        "dbp_mmhg":    "DBP (mmHg)",
        "sqi":         "SQI",
        "snr_db":      "SNR (dB)",
        "duration_s":  "Duration (s)",
        "method":      "Method",
    }
    show_df = df[[c for c in col_map if c in df.columns]].copy().rename(columns=col_map)
    show_df["Recorded At"] = show_df["Recorded At"].dt.strftime("%Y-%m-%d %H:%M")
    for col in ["HR (BPM)", "SpO2 (%)", "RR (br/min)", "SBP (mmHg)",
                "DBP (mmHg)", "SQI", "SNR (dB)"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    csv_bytes = show_df.to_csv(index=False).encode()
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"equiphys_{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ── All Patients page (doctor) ────────────────────────────────────────────────

def all_patients_page() -> None:
    import pandas as pd
    _page_header("All Patients", "Patient registry — doctor view")

    patients = get_all_patients_for_doctor()
    if not patients:
        st.info("No patients registered yet.")
        return

    df = pd.DataFrame(patients)[
        ["full_name", "username", "date_of_birth", "sex",
         "session_count", "updated_at"]
    ].rename(columns={
        "full_name":     "Name",
        "username":      "Username",
        "date_of_birth": "DOB",
        "sex":           "Sex",
        "session_count": "Sessions",
        "updated_at":    "Last Updated",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown('<div class="eq-section-label">View Patient</div>',
                unsafe_allow_html=True)
    names  = [f"{p['full_name']}  (@{p['username']})" for p in patients]
    chosen = st.selectbox("Select patient", ["— select —"] + names)
    if chosen and chosen != "— select —":
        idx = names.index(chosen)
        pid = patients[idx]["patient_id"]
        tab_prof, tab_hist = st.tabs(["Profile", "History"])
        with tab_prof:
            profile_page(read_only=True, patient_id=pid)
        with tab_hist:
            history_page(patient_id=pid)