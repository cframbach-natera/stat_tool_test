
import io, json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plots & analysis
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, pairwise_logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter

import plotly.express as px

st.set_page_config(page_title="QS Stats Lab (POC v4.1)", page_icon="🧪", layout="wide")

# ---------- Theming ----------
CUSTOM_CSS = """
<style>
.block-container { padding-top: 0.6rem; padding-bottom: 2rem; }
div[data-testid="stSidebar"] { min-width: 370px; }
.kpi-card {
  border-radius: 16px;
  padding: 12px 16px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.06);
  background: linear-gradient(180deg, rgba(250,250,255,1) 0%, rgba(245,245,250,1) 100%);
  border: 1px solid rgba(0,0,0,0.06);
}
.kpi-title { font-weight: 600; margin-bottom: 2px; }
.kpi-value { font-size: 1.1rem; font-weight: 600; }
.small { font-size: 0.85rem; color: #6b6f76; }
.at-risk-note { font-size: 0.8rem; color: #6b6f76; margin-top: -8px; }
hr.subtle { border: none; height: 1px; background: linear-gradient(90deg, rgba(0,0,0,0.0), rgba(0,0,0,0.08), rgba(0,0,0,0.0)); }
.panel {
  padding: 12px 14px; border-radius: 12px; border: 1px solid rgba(0,0,0,0.08);
  background: #fafbff;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Sidebar ----------
analysis = st.sidebar.radio(
    "Analysis",
    ["Kaplan–Meier (CKD cohort builder)", "Co-occurrence"],
    help="Select which analysis to run.",
    key="analysis_choice"
)

# ---------- Data ----------
def load_sample_ckd() -> pd.DataFrame:
    return pd.read_csv("sample_ckd.csv", parse_dates=["index_date","last_followup_date","death_date",
                                                      "ckd_stage1_date","ckd_stage2_date","ckd_stage3_date",
                                                      "ckd_stage4_date","ckd_esrd_date"])

def file_section(label: str):
    with st.sidebar.expander("📤 Data", expanded=True):
        use_sample = st.toggle(
            "Use sample CKD dataset",
            value=True if analysis.startswith("Kaplan") else False,
            help="Try the built-in CKD sample if you don't have a file handy.",
            key="use_sample_toggle"
        )
        file = st.file_uploader(label, type=["csv"], accept_multiple_files=False, key="csv_uploader")
        df = None
        if use_sample and file is None:
            df = load_sample_ckd()
            buf = io.StringIO(); df.to_csv(buf, index=False)
            st.download_button("Download sample CSV", data=buf.getvalue().encode("utf-8"),
                               file_name="sample_ckd.csv", mime="text/csv", key="dl_sample")
        elif file is not None:
            df = pd.read_csv(file)
            parse_cols = ["index_date","last_followup_date","death_date",
                          "ckd_stage1_date","ckd_stage2_date","ckd_stage3_date","ckd_stage4_date","ckd_esrd_date"]
            for c in parse_cols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

df = file_section("Upload CKD patient-level CSV")

# ---------- Presets ----------
ALLOWED_KEYS_STATIC = [
    "analysis_choice",
    "km_event_type","km_time_unit",
    "km_baseline_choice","km_prog_baseline_stages","km_prog_target_stage_min",
]

def preset_sidebar(df: pd.DataFrame):
    with st.sidebar.expander("⭐ Presets (beta)", expanded=False):
        st.caption("Save your current settings and filters to a JSON preset, or load a preset.")
        up = st.file_uploader("Load preset JSON", type=["json"], key="preset_upload")
        if up is not None:
            try:
                preset = json.loads(up.getvalue().decode("utf-8"))
                sess = preset.get("session", {})
                for k, v in sess.items():
                    st.session_state[k] = v
                st.success("Preset loaded. Refreshing with your saved settings.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load preset: {e}")
        st.markdown("---")
        name = st.text_input("Preset name", value=st.session_state.get("preset_name","my_ckd_preset"), key="preset_name")
        if st.button("Generate preset", type="primary", key="btn_make_preset"):
            preset = build_preset(df, name=name)
            st.session_state["last_preset_json"] = json.dumps(preset, indent=2)
        if "last_preset_json" in st.session_state:
            st.download_button("Download preset JSON",
                               data=st.session_state["last_preset_json"].encode("utf-8"),
                               file_name=f"qs_stats_ckd_preset_{name}.json",
                               mime="application/json",
                               key="dl_preset")

def build_preset(df: pd.DataFrame, name: str = "preset") -> Dict:
    dyn_keys = [k for k in st.session_state.keys() if k.startswith("A_") or k.startswith("B_") or k.startswith("flt_")]
    keep = {k: st.session_state[k] for k in ALLOWED_KEYS_STATIC if k in st.session_state}
    keep.update({k: st.session_state[k] for k in dyn_keys if k in st.session_state})
    return {"meta":{"name":name,"created":datetime.utcnow().isoformat()+"Z","columns":df.columns.tolist()},
            "session": keep}

if df is not None and not df.empty:
    preset_sidebar(df)

if df is None or df.empty:
    st.info("Upload a CSV or enable the sample dataset in the sidebar to continue.")
    st.stop()

# ---------- Helpers ----------
def is_binary_series(s: pd.Series) -> bool:
    vals = pd.Series(s.dropna().unique())
    if len(vals) == 0: return False
    normalized = (vals.replace({True:1, False:0}).astype(str).str.lower().replace({"true":"1","false":"0"}))
    return set(normalized.unique()).issubset({"0","1"})

def normalize_binary_series(s: pd.Series) -> pd.Series:
    return (s.replace({True:1, False:0}).astype(str).str.lower().replace({"true":"1","false":"0"}).fillna("0").astype(float).astype(int))

def render_kpis(summary_rows: List[Dict]):
    cols = st.columns(len(summary_rows))
    for i, r in enumerate(summary_rows):
        med = r.get("Median", np.nan)
        med_txt = "NA" if (med is None or (isinstance(med, float) and not np.isfinite(med))) else f"{med:.2f}"
        n = r.get("N", 0); e = r.get("Events", 0)
        rate = f"{(e/n*100):.0f}%" if n else "—"
        html = f"""
        <div class="kpi-card">
          <div class="kpi-title">{r.get("Label","Cohort")}</div>
          <div class="kpi-value">Median: {med_txt}</div>
          <div class="small">N={n} • Events={e} ({rate})</div>
        </div>
        """
        with cols[i]: st.markdown(html, unsafe_allow_html=True)

# ---------- Event mapping ----------
def km_event_mapper_ui(df: pd.DataFrame) -> Dict:
    st.markdown("### 1) Event Definition (KM)")
    with st.container():
        ev_type = st.radio("Event type", ["CKD progression", "Mortality", "Composite: progression≥X OR death"], index=0, key="km_event_type")

        baseline_choice = st.radio("Baseline (Day 0)",
                                   ["Index date", "Earliest baseline-stage date"],
                                   index=0 if st.session_state.get("km_baseline_choice","Index date")=="Index date" else 1,
                                   key="km_baseline_choice",
                                   help="If you choose 'Earliest baseline-stage date', Day 0 = min of the selected baseline stages' first dates."
                                   )

        time_unit = st.radio("Time unit", ["days","months","years"],
                             index={"days":0,"months":1,"years":2}[st.session_state.get("km_time_unit","months")],
                             key="km_time_unit")

        st.caption("Tip: In claims/EHR, index_date is often the study entry (e.g., first qualifying CKD record). Baseline-stage date is the earliest date the patient reached the chosen stage(s).")

        stage_cols = [c for c in df.columns if c.startswith("ckd_stage") or c=="ckd_esrd_date"]
        baseline = st.multiselect("Baseline stage(s) (used when baseline = baseline-stage)", options=[1,2,3],
                                  default=st.session_state.get("km_prog_baseline_stages",[1,2]), key="km_prog_baseline_stages")
        targ_min = st.selectbox("Progress to at least stage", options=[2,3,4,5], index=2, key="km_prog_target_stage_min", help="5 = ESRD")

    return {"type": ev_type, "unit": time_unit,
            "baseline_choice": baseline_choice,
            "baseline": baseline, "target_min": targ_min}

def compute_baseline_dates(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    if cfg["baseline_choice"].startswith("Index"):
        return pd.to_datetime(df["index_date"], errors="coerce")
    # earliest across selected baseline stage dates
    baseline_cols = [f"ckd_stage{s}_date" for s in cfg["baseline"] if s in [1,2,3]]
    if not baseline_cols:
        baseline_cols = ["ckd_stage1_date","ckd_stage2_date"]
    base_dates = pd.concat([pd.to_datetime(df[c], errors="coerce") for c in baseline_cols if c in df.columns], axis=1)
    return base_dates.min(axis=1)

def compute_target_progression_date(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    target_cols = []
    if cfg["target_min"] <= 2: target_cols += ["ckd_stage2_date"]
    if cfg["target_min"] <= 3: target_cols += ["ckd_stage3_date"]
    if cfg["target_min"] <= 4: target_cols += ["ckd_stage4_date"]
    target_cols += ["ckd_esrd_date"]  # always include ESRD
    tdates = pd.concat([pd.to_datetime(df[c], errors="coerce") for c in target_cols if c in df.columns], axis=1)
    return tdates.min(axis=1)

def compute_km_time_event(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    d = df.copy()
    base = compute_baseline_dates(d, cfg)
    death = pd.to_datetime(d["death_date"], errors="coerce")
    censor = pd.to_datetime(d["last_followup_date"], errors="coerce")

    if cfg["type"] == "Mortality":
        # event = death after baseline
        event_flag = (~death.isna()) & (~base.isna()) & (death > base)
        t_end = np.where(event_flag, death, censor)
    elif cfg["type"].startswith("Composite"):
        prog = compute_target_progression_date(d, cfg)
        # candidate event is earliest of (progression, death) that occurs AFTER baseline
        prog_valid = (~prog.isna()) & (~base.isna()) & (prog > base)
        death_valid = (~death.isna()) & (~base.isna()) & (death > base)
        # Use min where both exist, else whichever exists
        candidates = pd.concat([prog.where(prog_valid), death.where(death_valid)], axis=1).min(axis=1)
        event_flag = ~candidates.isna()
        t_end = np.where(event_flag, candidates, censor)
    else:  # CKD progression
        prog = compute_target_progression_date(d, cfg)
        event_flag = (~prog.isna()) & (~base.isna()) & (prog > base)
        t_end = np.where(event_flag, prog, censor)

    t_end = pd.to_datetime(t_end)
    # compute time deltas
    time_days = (t_end - base).dt.days.astype("float")
    d["km_time_days"] = time_days
    d["km_event"] = event_flag.astype(int)

    # Time unit conversion
    if cfg["unit"] == "months":
        d["km_time"] = d["km_time_days"]/30.4375
        d["km_time_label"] = "Months"
    elif cfg["unit"] == "years":
        d["km_time"] = d["km_time_days"]/365.25
        d["km_time_label"] = "Years"
    else:
        d["km_time"] = d["km_time_days"]
        d["km_time_label"] = "Days"

    # Drop rows with invalid/negative or missing times
    valid = (~d["km_time"].isna()) & np.isfinite(d["km_time"]) & (d["km_time"] >= 0)
    d = d[valid].copy()
    return d

# ---------- Cohort builder ----------
def build_filters_for(prefix: str, df: pd.DataFrame) -> Dict:
    st.markdown(f"#### Cohort {prefix}")
    filters = {}
    label_key = f"{prefix}_label"
    label = st.text_input("Cohort label", value=st.session_state.get(label_key, f"Cohort {prefix}"), key=label_key)

    # Proteinuria slider
    prot_col = None
    for cand in ["uacr_mg_g","proteinuria_value"]:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            prot_col = cand; break
    if prot_col is not None:
        min_v, max_v = float(np.nanmin(df[prot_col])), float(np.nanmax(df[prot_col]))
        key = f"{prefix}_proteinuria_range"
        default = st.session_state.get(key, (float(min_v), float(max_v)))
        rng = st.slider(f"{prot_col} range", min_value=float(min_v), max_value=float(max_v),
                        value=default, key=key)
        filters[prot_col] = ("range", rng)

    # General filters
    with st.expander("More filters", expanded=False):
        for col in df.columns:
            if col in ["km_time","km_time_days","km_event","km_time_label","baseline_date","target_date"]:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                _min, _max = float(np.nanmin(df[col])), float(np.nanmax(df[col]))
                if np.isfinite(_min) and np.isfinite(_max) and _min != _max:
                    key = f"{prefix}_flt_{col}_range"
                    default_val = st.session_state.get(key, (float(_min), float(_max)))
                    val = st.slider(f"{col} range", min_value=float(_min), max_value=float(_max),
                                    value=default_val, key=key)
                    filters[col] = ("range", val)
            elif pd.api.types.is_bool_dtype(df[col]) or is_binary_series(df[col]):
                key = f"{prefix}_flt_{col}_bin"
                default_choice = st.session_state.get(key, "All")
                choice = st.selectbox(f"{col} filter", ["All", "1 only", "0 only"],
                                      index=["All","1 only","0 only"].index(default_choice) if default_choice in ["All","1 only","0 only"] else 0, key=key)
                if choice != "All":
                    filters[col] = ("bin", 1 if choice=="1 only" else 0)
            else:
                nunique = df[col].nunique(dropna=True)
                if 1 < nunique <= 50:
                    opts = sorted([x for x in df[col].dropna().unique().tolist()], key=lambda x: str(x))
                    key = f"{prefix}_flt_{col}_in"
                    default_sel = st.session_state.get(key, opts)
                    sel = st.multiselect(f"{col} is any of", opts, default=default_sel, key=key)
                    if len(sel) != len(opts):
                        filters[col] = ("in", set(sel))

    built = st.button(f"Build {label}", key=f"{prefix}_build", type="primary")
    return {"label": label, "filters": filters, "built": built}

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, (ftype, val) in filters.items():
        if ftype == "range":
            lo, hi = val; mask &= df[col].astype(float).between(lo, hi)
        elif ftype == "bin":
            mask &= (normalize_binary_series(df[col]) == int(val))
        elif ftype == "in":
            mask &= df[col].isin(val)
    return df[mask].copy()

# ---------- KM page ----------
def km_ckd_page(df: pd.DataFrame):
    st.header("Kaplan–Meier (CKD Cohort Builder)")

    # 1) Event Definition
    cfg = km_event_mapper_ui(df)
    df_mapped = compute_km_time_event(df, cfg)

    # Inform about dropped rows (missing baseline or invalid times)
    dropped = len(df) - len(df_mapped)
    if dropped > 0:
        st.caption(f"Note: {dropped} rows dropped due to missing baseline, invalid event/censor dates, or negative durations.")

    st.markdown("---")
    st.markdown("### 2) Build two cohorts")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        cohortA = build_filters_for("A", df_mapped)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        cohortB = build_filters_for("B", df_mapped)
        st.markdown('</div>', unsafe_allow_html=True)

    if not (cohortA["built"] and cohortB["built"]):
        st.info("Define and **Build** both cohorts to render the KM comparison.")
        return

    dfA = apply_filters(df_mapped, cohortA["filters"])
    dfB = apply_filters(df_mapped, cohortB["filters"])

    for name, d in [("A", dfA), ("B", dfB)]:
        if d.empty:
            st.warning(f"Cohort {name} is empty after filters.")
    if dfA.empty or dfB.empty:
        return

    # KPI cards
    kpis = []
    for lab, d in [(cohortA["label"], dfA), (cohortB["label"], dfB)]:
        try:
            km = KaplanMeierFitter().fit(d["km_time"], d["km_event"])
            med = float(km.median_survival_time_) if km.median_survival_time_ is not None else np.nan
        except Exception:
            med = np.nan
        kpis.append({"Label": lab, "N": int(len(d)), "Events": int(d["km_event"].sum()), "Median": med})
    render_kpis(kpis)
    st.markdown('<hr class="subtle" />', unsafe_allow_html=True)

    # Plot
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fitters = []
    for lab, d in [(cohortA["label"], dfA), (cohortB["label"], dfB)]:
        km = KaplanMeierFitter()
        try:
            km.fit(durations=d["km_time"], event_observed=d["km_event"], label=str(lab))
            km.plot_survival_function(ax=ax, ci_show=True)
            fitters.append(km)
        except Exception as e:
            st.warning(f"Could not fit KM for cohort '{lab}': {e}")
    ax.set_xlabel(f"Time ({dfA['km_time_label'].iloc[0] if len(dfA) else 'time'})")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    try:
        if len(fitters) >= 1:
            add_at_risk_counts(*fitters, ax=ax)
            st.markdown('<div class="at-risk-note">At-risk counts shown below the x-axis.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.info(f"At-risk table unavailable: {e}")
    st.pyplot(fig, use_container_width=True)

    # Stats
    with st.expander("📊 Statistics"):
        if len(dfA) >= 1 and len(dfB) >= 1:
            try:
                res = logrank_test(dfA["km_time"], dfB["km_time"], event_observed_A=dfA["km_event"], event_observed_B=dfB["km_event"])
                st.write(f"Log-rank test: p = **{res.p_value:.4g}**")
            except Exception as e:
                st.info(f"Log-rank failed: {e}")
            # Cox HR (binary)
            try:
                tmp = pd.concat([dfA.assign(group=0), dfB.assign(group=1)], axis=0, ignore_index=True)
                cph = CoxPHFitter()
                cph.fit(tmp[["km_time","km_event","group"]], duration_col="km_time", event_col="km_event")
                hr = float(np.exp(cph.params_["group"]))
                ci = cph.confidence_intervals_.loc["group"].values
                st.write(f"Cox PH HR (B vs A): **{hr:.3g}**  [{np.exp(ci[0]):.3g}, {np.exp(ci[1]):.3g}]")
            except Exception as e:
                st.info(f"Cox PH failed: {e}")
            # Pairwise log-rank
            try:
                pair = pairwise_logrank_test(
                    pd.concat([dfA["km_time"], dfB["km_time"]]).values,
                    groups=np.array([0]*len(dfA)+[1]*len(dfB)),
                    event_observed=pd.concat([dfA["km_event"], dfB["km_event"]]).values,
                    p_adjust_method="holm"
                )
                st.write("Pairwise log-rank (Holm-adjusted):")
                st.dataframe(pair.summary.round(4))
            except Exception:
                pass

    # Export
    with st.expander("📥 Export"):
        out = pd.concat([dfA.assign(__cohort=cohortA["label"]), dfB.assign(__cohort=cohortB["label"])], ignore_index=True)
        cols = ["__cohort","km_time","km_event","km_time_label","index_date","last_followup_date","death_date"]
        for c in ["ckd_stage1_date","ckd_stage2_date","ckd_stage3_date","ckd_stage4_date","ckd_esrd_date"]:
            if c in out.columns: cols.append(c)
        csv = out[cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download KM analysis dataset (CSV)", data=csv, file_name="km_ckd_analysis.csv", mime="text/csv")

# ---------- Route ----------
if analysis.startswith("Kaplan"):
    km_ckd_page(df)
else:
    st.write("Co-occurrence unchanged in this v4.1 POC. Switch back to Kaplan–Meier to try the CKD cohort builder.")

st.sidebar.caption("POC v4.1 • CKD-focused KM cohort builder • Mortality baseline toggle + Composite endpoint.")
