
import io, json, hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib as mpl
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter

st.set_page_config(page_title="QS Oncology Stats (POC v5.1)", page_icon="🧬", layout="wide")

# ---------- Theme palette ----------
PALETTE = {
    "blue": "#60A4BF",
    "green": "#7EB86E",
    "blue100": "#CEDFFB",
    "blue200": "#A9CCF5",
    "blue500": "#329ADB",
    "blue900": "#1B5F7F",
    "gray": "#8D97A1",
}

mpl.rcParams.update({
    "axes.edgecolor": PALETTE["blue900"],
    "axes.labelcolor": PALETTE["blue900"],
    "xtick.color": PALETTE["blue900"],
    "ytick.color": PALETTE["blue900"],
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.color": PALETTE["gray"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": PALETTE["blue900"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12.0,
    "axes.titleweight": "semibold",
})

CUSTOM_CSS = f"""
<style>
:root {{
  --qs-blue: {PALETTE['blue']};
  --qs-green: {PALETTE['green']};
  --qs-blue100: {PALETTE['blue100']};
  --qs-blue200: {PALETTE['blue200']};
  --qs-blue500: {PALETTE['blue500']};
  --qs-blue900: {PALETTE['blue900']};
  --qs-gray: {PALETTE['gray']};
}}
html, body, [class*="css"]  {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji";
}}
.block-container {{ padding-top: 0.5rem; padding-bottom: 2rem; }}
.header-bar {{
  position: sticky; top: 0; z-index: 2;
  display:flex; align-items:center; gap:12px;
  padding: 12px 16px; border-radius: 14px;
  background: var(--qs-blue100);
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 2px 16px rgba(0,0,0,0.06);
  margin-bottom: 12px;
}}
.header-title {{ font-weight: 800; color: var(--qs-blue900); font-size: 1.1rem; }}
.header-sub {{ color: var(--qs-blue900); font-size: 0.9rem; font-weight: 500;}}
.panel {{ padding: 12px 14px; border-radius: 14px; border: 1px solid var(--qs-blue100); background: #ffffff; box-shadow: 0 2px 12px rgba(0,0,0,0.04); }}
.kpi-card {{ border-radius: 16px; padding: 12px 16px; box-shadow: 0 2px 16px rgba(0,0,0,0.06); background: #ffffff; border: 1px solid var(--qs-blue100); }}
.kpi-title {{ font-weight: 600; margin-bottom: 2px; color: var(--qs-blue900); }}
.kpi-value {{ font-size: 1.1rem; font-weight: 700; color: var(--qs-blue500); }}
.small {{ font-size: 0.85rem; color: var(--qs-gray); }}
hr.subtle {{ border: none; height: 1px; background: linear-gradient(90deg, rgba(0,0,0,0.0), rgba(0,0,0,0.12), rgba(0,0,0,0.0)); }}
.condition-card {{ border: 1px dashed var(--qs-blue200); border-radius: 12px; padding: 10px; margin-bottom: 8px; }}
.stButton > button {{ background-color: var(--qs-blue500); border: 1px solid var(--qs-blue900); color: white; font-weight: 600; border-radius: 10px; }}
.stButton > button:hover {{ filter: brightness(0.98); }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Tabs ----------
tabs = st.tabs(["📁 Data", "👥 Cohorts", "🎯 Endpoint & Model", "📈 KM & Stats"])

# ---------- Data loaders ----------
PATIENT_ANCHORS = [
    "diagnosis_date","surgery_date","first_treatment_date","progression_date","next_treatment_date","death_date","last_followup_date"
]

def load_patients_sample() -> pd.DataFrame:
    return pd.read_csv("patients_sample.csv", parse_dates=PATIENT_ANCHORS)

def load_tests_sample() -> pd.DataFrame:
    return pd.read_csv("tests_sample.csv", parse_dates=["test_date"])

def data_tab():
    with tabs[0]:
        st.markdown('<div class="header-bar"><div class="header-title">Data</div><div class="header-sub">Upload patients & tests, or use the samples.</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            use_pat_sample = st.toggle("Use patients sample", value=True, key="use_pat_sample")
            pat_file = st.file_uploader("Upload patients.csv", type=["csv"], key="pat_upl")
            dfp = None
            if use_pat_sample and pat_file is None:
                dfp = load_patients_sample()
                st.download_button("Download patients sample", data=dfp.to_csv(index=False).encode("utf-8"), file_name="patients_sample.csv", mime="text/csv")
            elif pat_file is not None:
                dfp = pd.read_csv(pat_file, parse_dates=PATIENT_ANCHORS, infer_datetime_format=True)
        with c2:
            use_tests_sample = st.toggle("Use tests sample", value=True, key="use_tests_sample")
            tst_file = st.file_uploader("Upload tests.csv", type=["csv"], key="tst_upl")
            dft = None
            if use_tests_sample and tst_file is None:
                dft = load_tests_sample()
                st.download_button("Download tests sample", data=dft.to_csv(index=False).encode("utf-8"), file_name="tests_sample.csv", mime="text/csv")
            elif tst_file is not None:
                dft = pd.read_csv(tst_file, parse_dates=["test_date"], infer_datetime_format=True)

        st.session_state["patients_df"] = dfp
        st.session_state["tests_df"] = dft
        if dfp is not None and dft is not None:
            st.success(f"Loaded {len(dfp):,} patients and {len(dft):,} tests.")
            with st.expander("Preview patients (top 20)"):
                st.dataframe(dfp.head(20))
            with st.expander("Preview tests (top 20)"):
                st.dataframe(dft.head(20))
        else:
            st.info("Load both patients and tests to continue.")

# ---------- Cohorts: quick filters ----------
QUICK_DEMO_COLS = {
    "age_at_diag": ("range", int),
    "sex": ("in", str),
    "race": ("in", str),
    "ethnicity": ("in", str),
    "stage_group": ("in", str),
    "cancer_type": ("in", str),
}

def render_quick_filters(prefix: str, dfp: pd.DataFrame) -> Dict:
    st.subheader(f"Cohort {prefix} – quick filters")
    filters: Dict = {}
    label_key = f"{prefix}_label"; label = st.text_input("Cohort label", value=st.session_state.get(label_key, f"Cohort {prefix}"), key=label_key)

    c1, c2, c3 = st.columns(3)
    # age range
    if "age_at_diag" in dfp.columns:
        a_min, a_max = int(np.nanmin(dfp["age_at_diag"])), int(np.nanmax(dfp["age_at_diag"]))
        key = f"{prefix}_age_range"
        default = st.session_state.get(key, (a_min, a_max))
        with c1:
            rng = st.slider("Age at diagnosis", min_value=a_min, max_value=a_max, value=(default[0], default[1]), key=key)
        filters["age_at_diag"] = ("range", rng)

    # sex, race, stage
    for col, col_title, col_container in [
        ("sex","Biological sex", c1),
        ("race","Race", c2),
        ("ethnicity","Ethnicity", c2),
        ("stage_group","Stage group", c3),
        ("cancer_type","Cancer type", c3),
    ]:
        if col in dfp.columns:
            opts = sorted([x for x in dfp[col].dropna().unique().tolist()], key=lambda x: str(x))
            key = f"{prefix}_{col}_in"
            default = st.session_state.get(key, opts)
            with col_container:
                sel = st.multiselect(col_title, options=opts, default=default, key=key)
            if len(sel) != len(opts):
                filters[col] = ("in", set(sel))

    return label, filters

# ---------- Rule Builder (tests-based) ----------
RESULT_VALUES = ["positive","negative","indeterminate"]
FACET_COLS = ["assay_version","specimen_type","lab_site"]

def init_rules(prefix: str):
    st.session_state.setdefault(f"{prefix}_rules", [])
    st.session_state.setdefault(f"{prefix}_rules_logic", "ALL")

def add_condition(prefix: str):
    default_cond = {
        "type": "Count",  # Count, Exists, None, FirstResult, LastResult, Consecutive, Proportion
        "start_anchor": "surgery_date",
        "start_offset": 0,
        "end_mode": "anchor",  # anchor or offset
        "end_anchor": "first_treatment_date",
        "end_offset": 90,
        "include_start": True,
        "include_end": True,
        "results": ["positive","negative"],  # used for Count/Exists/None/Consecutive/Proportion
        "require_result": "positive",        # used for FirstResult/LastResult
        "include_indeterminate": False,
        "count_op": "at least",  # at least / at most / exactly
        "count_n": 1,
        "consecutive_k": 2,
        "proportion_pct": 50,
        "min_tests": 1,
        "facet_filters": {},  # {"assay_version": ["v1","v2"], ...}
    }
    st.session_state[f"{prefix}_rules"].append(default_cond)

def render_condition(prefix: str, idx: int, tests_df: pd.DataFrame):
    cond = st.session_state[f"{prefix}_rules"][idx]
    st.markdown('<div class="condition-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2,1,1])
    with c1:
        cond["type"] = st.selectbox("Condition type", ["Count","Exists","None","FirstResult","LastResult","Consecutive","Proportion"],
                                    index=["Count","Exists","None","FirstResult","LastResult","Consecutive","Proportion"].index(cond["type"]), key=f"{prefix}_type_{idx}")
    with c2:
        cond["start_anchor"] = st.selectbox("Start anchor", PATIENT_ANCHORS, index=PATIENT_ANCHORS.index(cond["start_anchor"]), key=f"{prefix}_sa_{idx}")
        cond["start_offset"] = st.number_input("Start offset (days)", value=int(cond["start_offset"]), step=1, key=f"{prefix}_so_{idx}")
    with c3:
        cond["end_mode"] = st.radio("End by", ["anchor","offset"], index=0 if cond["end_mode"]=="anchor" else 1, horizontal=True, key=f"{prefix}_em_{idx}")
        if cond["end_mode"] == "anchor":
            cond["end_anchor"] = st.selectbox("End anchor", PATIENT_ANCHORS, index=PATIENT_ANCHORS.index(cond["end_anchor"]), key=f"{prefix}_ea_{idx}")
        else:
            cond["end_offset"] = st.number_input("End offset (days)", value=int(cond["end_offset"]), step=1, key=f"{prefix}_eo_{idx}")
    c4, c5 = st.columns(2)
    with c4:
        cond["include_start"] = st.checkbox("Include start day", value=bool(cond["include_start"]), key=f"{prefix}_is_{idx}")
    with c5:
        cond["include_end"] = st.checkbox("Include end day", value=bool(cond["include_end"]), key=f"{prefix}_ie_{idx}")

    if cond["type"] in ["FirstResult","LastResult"]:
        cond["require_result"] = st.selectbox("Required result", RESULT_VALUES, index=RESULT_VALUES.index(cond["require_result"]), key=f"{prefix}_req_{idx}")
        cond["include_indeterminate"] = st.checkbox("Consider indeterminate in ordering", value=bool(cond["include_indeterminate"]), key=f"{prefix}_ind_{idx}")
    elif cond["type"] in ["Count","Exists","None","Consecutive","Proportion"]:
        cond["results"] = st.multiselect("Test results to include", RESULT_VALUES, default=cond["results"], key=f"{prefix}_res_{idx}")
        cond["include_indeterminate"] = st.checkbox("Include indeterminate in pool", value=bool(cond["include_indeterminate"]), key=f"{prefix}_ind_{idx}")
        if cond["type"] == "Count":
            cond["count_op"] = st.selectbox("Comparator", ["at least","at most","exactly"], index=["at least","at most","exactly"].index(cond["count_op"]), key=f"{prefix}_cop_{idx}")
            cond["count_n"] = st.number_input("Count N", value=int(cond["count_n"]), min_value=0, step=1, key=f"{prefix}_cn_{idx}")
        if cond["type"] == "Consecutive":
            cond["consecutive_k"] = st.number_input("Consecutive K", value=int(cond["consecutive_k"]), min_value=1, step=1, key=f"{prefix}_ck_{idx}")
        if cond["type"] == "Proportion":
            cond["proportion_pct"] = st.number_input("Proportion threshold (%)", value=int(cond["proportion_pct"]), min_value=0, max_value=100, step=1, key=f"{prefix}_pp_{idx}")
            cond["min_tests"] = st.number_input("Minimum tests in window", value=int(cond["min_tests"]), min_value=0, step=1, key=f"{prefix}_mt_{idx}")

    # facet filters
    if tests_df is not None:
        with st.expander("Facet filters (optional)"):
            for fc in [c for c in FACET_COLS if c in tests_df.columns]:
                opts = sorted([x for x in tests_df[fc].dropna().unique().tolist()], key=lambda x: str(x))
                sel = st.multiselect(fc, options=opts, default=cond["facet_filters"].get(fc, opts), key=f"{prefix}_facet_{idx}_{fc}")
                if len(sel) != len(opts):
                    cond["facet_filters"][fc] = sel
                elif fc in cond["facet_filters"]:
                    del cond["facet_filters"][fc]

    # delete
    if st.button("Delete condition", key=f"{prefix}_del_{idx}"):
        st.session_state[f"{prefix}_rules"].pop(idx)
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def window_for_condition(patients: pd.DataFrame, cond: Dict) -> pd.DataFrame:
    # Build per-patient window [start, end]
    start = pd.to_datetime(patients[cond["start_anchor"]], errors="coerce") + pd.to_timedelta(int(cond["start_offset"]), unit="D")
    if cond["end_mode"] == "anchor":
        end = pd.to_datetime(patients[cond["end_anchor"]], errors="coerce")
    else:
        end = start + pd.to_timedelta(int(cond["end_offset"]), unit="D")
    return pd.DataFrame({"patient_id": patients["patient_id"], "win_start": start, "win_end": end})

def filter_tests_in_window(tests: pd.DataFrame, win: pd.DataFrame, cond: Dict) -> pd.DataFrame:
    d = tests.merge(win, on="patient_id", how="left")
    if cond["include_start"]:
        left_ok = d["test_date"] >= d["win_start"]
    else:
        left_ok = d["test_date"] > d["win_start"]
    if cond["include_end"]:
        right_ok = (d["win_end"].isna()) | (d["test_date"] <= d["win_end"])
    else:
        right_ok = (d["win_end"].isna()) | (d["test_date"] < d["win_end"])
    m = left_ok & right_ok
    d = d[m].copy()

    # Result filtering for certain types
    if cond["type"] in ["Count","Exists","None","Consecutive","Proportion"]:
        if cond["results"]:
            d = d[d["result"].isin(cond["results"] + (["indeterminate"] if cond["include_indeterminate"] and "indeterminate" not in cond["results"] else []))]
        elif not cond.get("include_indeterminate", False):
            d = d[d["result"].isin(["positive","negative"])]  # exclude indeterminate by default

    # facet filters
    for fc, vals in cond.get("facet_filters", {}).items():
        if fc in d.columns:
            d = d[d[fc].isin(vals)]
    return d

def eval_condition(patients: pd.DataFrame, tests: pd.DataFrame, cond: Dict) -> pd.Series:
    # Return boolean Series indexed like patients, True if patient satisfies condition
    win = window_for_condition(patients, cond)
    d = filter_tests_in_window(tests, win, cond)
    grouped = d.groupby("patient_id")

    satisfied = pd.Series(False, index=patients.index)

    if cond["type"] == "Exists":
        has_any = grouped.size().rename("cnt")
        satisfied = patients["patient_id"].map(has_any.gt(0)).fillna(False)

    elif cond["type"] == "None":
        has_any = grouped.size().rename("cnt")
        satisfied = ~patients["patient_id"].map(has_any.gt(0)).fillna(False)

    elif cond["type"] == "Count":
        counts = grouped.size().rename("cnt")
        op = cond["count_op"]; n = int(cond["count_n"])
        if op == "at least":
            ok = counts.ge(n)
        elif op == "at most":
            ok = counts.le(n)
        else:
            ok = counts.eq(n)
        satisfied = patients["patient_id"].map(ok).fillna(False)

    elif cond["type"] in ["FirstResult","LastResult"]:
        # Consider ordering; optionally include indeterminate in ordering
        dd = tests.merge(win, on="patient_id", how="left")
        if cond["include_start"]:
            left_ok = dd["test_date"] >= dd["win_start"]
        else:
            left_ok = dd["test_date"] > dd["win_start"]
        if cond["include_end"]:
            right_ok = (dd["win_end"].isna()) | (dd["test_date"] <= dd["win_end"])
        else:
            right_ok = (dd["win_end"].isna()) | (dd["test_date"] < dd["win_end"])
        m = left_ok & right_ok
        dd = dd[m].copy()
        if not cond["include_indeterminate"]:
            dd = dd[dd["result"] != "indeterminate"]
        order = dd.sort_values(["patient_id","test_date"])
        if cond["type"] == "FirstResult":
            pick = order.groupby("patient_id").first()["result"]
        else:
            pick = order.groupby("patient_id").last()["result"]
        required = cond["require_result"]
        ok = pick.eq(required)
        satisfied = patients["patient_id"].map(ok).fillna(False)

    elif cond["type"] == "Consecutive":
        k = int(cond["consecutive_k"])
        if k <= 1:
            has_any = grouped.size().rename("cnt").gt(0)
            satisfied = patients["patient_id"].map(has_any).fillna(False)
        else:
            ok_ids = set()
            for pid, g in grouped:
                g = g.sort_values("test_date")
                seq = g["result"].tolist()
                # Convert to 1 if in selected results, else 0
                sel = set(cond.get("results", []))
                if cond.get("include_indeterminate", False):
                    sel = sel.union({"indeterminate"})
                arr = [1 if r in sel else 0 for r in seq]
                # check for run length >= k
                run = 0
                for v in arr:
                    if v == 1:
                        run += 1
                        if run >= k:
                            ok_ids.add(pid)
                            break
                    else:
                        run = 0
            satisfied = patients["patient_id"].isin(ok_ids)

    elif cond["type"] == "Proportion":
        pct = float(cond["proportion_pct"])
        min_tests = int(cond.get("min_tests", 1))
        cnts = grouped.size().rename("n_total")
        # mark "in" selection
        sel = set(cond.get("results", []))
        if cond.get("include_indeterminate", False):
            sel = sel.union({"indeterminate"})
        in_sel = d[d["result"].isin(sel)]
        cnt_in = in_sel.groupby("patient_id").size().rename("n_in")
        joined = pd.concat([cnts, cnt_in], axis=1).fillna(0)
        prop = (joined["n_in"] / joined["n_total"]).where(joined["n_total"] >= min_tests, 0.0)
        ok = prop.ge(pct / 100.0)
        satisfied = patients["patient_id"].map(ok).fillna(False)

    return satisfied

def apply_rules(prefix: str, patients: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    rules: List[Dict] = st.session_state.get(f"{prefix}_rules", [])
    logic = st.session_state.get(f"{prefix}_rules_logic","ALL")
    if not rules:
        return patients
    masks = []
    for cond in rules:
        m = eval_condition(patients, tests, cond)
        masks.append(m)
    if logic == "ALL":
        combined = masks[0].copy()
        for m in masks[1:]:
            combined &= m
    else:
        combined = masks[0].copy()
        for m in masks[1:]:
            combined |= m
    return patients[combined].copy()

def apply_quick_filters(patients: pd.DataFrame, qf: Dict) -> pd.DataFrame:
    d = patients.copy()
    mask = pd.Series(True, index=d.index)
    for col, (ftype, val) in qf.items():
        if ftype == "range":
            lo, hi = val; mask &= d[col].astype(float).between(lo, hi)
        elif ftype == "in":
            mask &= d[col].isin(val)
    return d[mask].copy()

def build_cohort_ui(prefix: str):
    with st.container():
        patients = st.session_state.get("patients_df")
        tests = st.session_state.get("tests_df")
        if patients is None or tests is None:
            st.info("Load patients and tests in the **Data** tab first."); return None

        label, qfilters = render_quick_filters(prefix, patients)

        # Rule builder
        st.markdown("### Rules (tests)")
        init_rules(prefix)
        # aggregator
        st.session_state[f"{prefix}_rules_logic"] = st.radio("Combine conditions with", ["ALL","ANY"], horizontal=True, key=f"{prefix}_agg")
        st.button("Add condition", on_click=add_condition, args=(prefix,), key=f"{prefix}_add")
        for i in range(len(st.session_state[f"{prefix}_rules"])):
            render_condition(prefix, i, tests)

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button(f"Build {label}", type="primary", key=f"{prefix}_build"):
                st.session_state[f"{prefix}_built"] = True
                st.session_state[f"{prefix}_label_built"] = label
                st.session_state[f"{prefix}_qf_built"] = qfilters
                st.session_state[f"{prefix}_rules_built"] = st.session_state[f"{prefix}_rules"]
                st.session_state[f"{prefix}_logic_built"] = st.session_state[f"{prefix}_rules_logic"]
                st.success(f"Cohort {prefix} built.")
        with col2:
            if st.button(f"Reset {prefix}", key=f"{prefix}_reset"):
                for k in [f"{prefix}_built", f"{prefix}_label_built", f"{prefix}_qf_built", f"{prefix}_rules_built", f"{prefix}_logic_built", f"{prefix}_rules"]:
                    if k in st.session_state: del st.session_state[k]
                init_rules(prefix)
                st.info(f"Cohort {prefix} reset.")

        built = st.session_state.get(f"{prefix}_built", False)
        label_built = st.session_state.get(f"{prefix}_label_built", label)
        qf_built = st.session_state.get(f"{prefix}_qf_built", qfilters)
        rules_built = st.session_state.get(f"{prefix}_rules_built", st.session_state.get(f"{prefix}_rules", []))
        logic_built = st.session_state.get(f"{prefix}_logic_built", st.session_state.get(f"{prefix}_rules_logic", "ALL"))
        return {"label": label_built, "qf": qf_built, "rules": rules_built, "logic": logic_built, "built": built}

def cohorts_tab():
    with tabs[1]:
        st.markdown('<div class="header-bar"><div class="header-title">Cohorts</div><div class="header-sub">Compose flexible test-based rules + quick filters.</div></div>', unsafe_allow_html=True)
        patients = st.session_state.get("patients_df"); tests = st.session_state.get("tests_df")
        if patients is None or tests is None:
            st.info("Load data in the **Data** tab first."); return
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            A = build_cohort_ui("A")
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            B = build_cohort_ui("B")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Reset both cohorts"):
            for p in ["A","B"]:
                for k in [f"{p}_built", f"{p}_label_built", f"{p}_qf_built", f"{p}_rules_built", f"{p}_logic_built", f"{p}_rules"]:
                    if k in st.session_state: del st.session_state[k]
            init_rules("A"); init_rules("B")
            st.info("Both cohorts reset.")

        st.session_state["cohortA"] = A; st.session_state["cohortB"] = B

# ---------- Endpoint & KM mapping ----------
def endpoint_tab():
    with tabs[2]:
        st.markdown('<div class="header-bar"><div class="header-title">Endpoint & Model</div><div class="header-sub">Choose endpoint and baseline for survival time.</div></div>', unsafe_allow_html=True)
        patients = st.session_state.get("patients_df")
        if patients is None:
            st.info("Load data in the **Data** tab first."); return

        ev_type = st.radio("Endpoint", [
            "Overall Survival (OS): death",
            "Time to Progression (TTP): progression only",
            "Progression-Free Survival (PFS): progression or death",
            "Time to Treatment (TTT): first treatment",
            "Time to Next Treatment (TTNT): next treatment"
        ], index=2, key="ev_type")

        baseline = st.radio("Baseline (Day 0)", [
            "Diagnosis date",
            "Positive test date (first)",
            "Surgery date",
            "First treatment date"
        ], index=0, key="baseline_choice")

        unit = st.radio("Time unit", ["days","months","years"], index=1, key="time_unit")
        st.session_state["endpoint_cfg"] = {"type": ev_type, "baseline": baseline, "unit": unit}
        st.success("Endpoint settings saved.")

def choose_baseline_series(patients: pd.DataFrame, cfg: Dict, tests: Optional[pd.DataFrame]) -> pd.Series:
    if cfg["baseline"].startswith("Diagnosis"):
        return pd.to_datetime(patients["diagnosis_date"], errors="coerce")
    if cfg["baseline"].startswith("Surgery"):
        return pd.to_datetime(patients["surgery_date"], errors="coerce")
    if cfg["baseline"].startswith("First treatment"):
        return pd.to_datetime(patients["first_treatment_date"], errors="coerce")
    # Positive test date (first): derive from tests
    if tests is None or tests.empty:
        return pd.Series(pd.NaT, index=patients.index)
    first_pos = tests[tests["result"]=="positive"].sort_values(["patient_id","test_date"]).groupby("patient_id").first()["test_date"]
    return patients["patient_id"].map(first_pos)

def compute_km_dataset(patients: pd.DataFrame, tests: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    d = patients.copy()
    base = choose_baseline_series(d, cfg, tests)

    death = pd.to_datetime(d["death_date"], errors="coerce")
    prog = pd.to_datetime(d["progression_date"], errors="coerce")
    next_tx = pd.to_datetime(d["next_treatment_date"], errors="coerce")
    first_tx = pd.to_datetime(d["first_treatment_date"], errors="coerce")
    censor = pd.to_datetime(d["last_followup_date"], errors="coerce")

    if cfg["type"].startswith("Overall Survival"):
        event = (~death.isna()) & (~base.isna()) & (death > base); t_end = np.where(event, death, censor)
    elif cfg["type"].startswith("Time to Progression"):
        event = (~prog.isna()) & (~base.isna()) & (prog > base); t_end = np.where(event, prog, censor)
    elif cfg["type"].startswith("Progression-Free Survival"):
        cand = pd.concat([prog, death], axis=1).min(axis=1)
        event = (~cand.isna()) & (~base.isna()) & (cand > base); t_end = np.where(event, cand, censor)
    elif cfg["type"].startswith("Time to Treatment (TTT)"):
        event = (~first_tx.isna()) & (~base.isna()) & (first_tx > base); t_end = np.where(event, first_tx, censor)
    else:  # TTNT
        event = (~next_tx.isna()) & (~base.isna()) & (next_tx > base); t_end = np.where(event, next_tx, censor)

    t_end = pd.to_datetime(t_end)
    time_days = (t_end - base).dt.days.astype("float")
    d["km_time_days"] = time_days
    d["km_event"] = event.astype(int)

    unit = cfg["unit"]
    if unit == "months":
        d["km_time"] = d["km_time_days"]/30.4375; d["km_time_label"] = "Months"
    elif unit == "years":
        d["km_time"] = d["km_time_days"]/365.25; d["km_time_label"] = "Years"
    else:
        d["km_time"] = d["km_time_days"]; d["km_time_label"] = "Days"

    valid = (~d["km_time"].isna()) & np.isfinite(d["km_time"]) & (d["km_time"] >= 0)
    return d[valid].copy()

# ---------- KM & Stats ----------
def km_stats_tab():
    with tabs[3]:
        st.markdown('<div class="header-bar"><div class="header-title">KM & Stats</div><div class="header-sub">Build cohorts first, then render curves + stats.</div></div>', unsafe_allow_html=True)
        patients = st.session_state.get("patients_df"); tests = st.session_state.get("tests_df")
        if patients is None or tests is None:
            st.info("Load data in the **Data** tab first."); return
        A = st.session_state.get("cohortA"); B = st.session_state.get("cohortB")
        if not A or not B or not (A["built"] and B["built"]):
            st.info("Build **both** cohorts in the Cohorts tab."); return
        cfg = st.session_state.get("endpoint_cfg")
        if not cfg:
            st.info("Set an endpoint in the **Endpoint & Model** tab."); return

        # Apply quick filters then rule sets
        dA_q = apply_quick_filters(patients, A["qf"]); dB_q = apply_quick_filters(patients, B["qf"])
        dA = apply_rules("A", dA_q, tests); dB = apply_rules("B", dB_q, tests)

        if dA.empty or dB.empty:
            st.warning("One of the cohorts is empty after filters/rules."); return

        mA = compute_km_dataset(dA, tests, cfg); mB = compute_km_dataset(dB, tests, cfg)

        # KPIs
        kpis = []
        for lab, d in [(A["label"], mA), (B["label"], mB)]:
            try:
                km = KaplanMeierFitter().fit(d["km_time"], d["km_event"])
                med = float(km.median_survival_time_) if km.median_survival_time_ is not None else np.nan
            except Exception:
                med = np.nan
            kpis.append({"Label": lab, "N": int(len(d)), "Events": int(d["km_event"].sum()), "Median": med})
        cols = st.columns(2)
        for i, r in enumerate(kpis):
            med_txt = "NA" if (r["Median"] is None or (isinstance(r["Median"], float) and not np.isfinite(r["Median"]))) else f"{r['Median']:.2f}"
            rate = f"{(r['Events']/r['N']*100):.0f}%" if r["N"] else "—"
            html = f"""
            <div class="kpi-card">
              <div class="kpi-title">{r['Label']}</div>
              <div class="kpi-value">Median: {med_txt}</div>
              <div class="small">N={r['N']} • Events={r['Events']} ({rate})</div>
            </div>
            """
            with cols[i]: st.markdown(html, unsafe_allow_html=True)
        st.markdown('<hr class="subtle" />', unsafe_allow_html=True)

        # Plot
        fig, ax = plt.subplots(figsize=(9.6, 5.8))
        fitters = []
        colors = [PALETTE["blue500"], PALETTE["green"]]
        for (lab, d), color in zip([(A["label"], mA), (B["label"], mB)], colors):
            km = KaplanMeierFitter()
            try:
                km.fit(durations=d["km_time"], event_observed=d["km_event"], label=str(lab))
                km.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.2)
                fitters.append(km)
            except Exception as e:
                st.warning(f"Could not fit KM for cohort '{lab}': {e}")
        ax.set_xlabel(f"Time ({mA['km_time_label'].iloc[0] if len(mA) else 'time'})", color=PALETTE["blue900"])
        ax.set_ylabel("Survival probability", color=PALETTE["blue900"])
        leg = ax.legend(framealpha=0.95, title=None)
        for text in leg.get_texts(): text.set_color(PALETTE["blue900"])
        try:
            if len(fitters) >= 1:
                add_at_risk_counts(*fitters, ax=ax, xticks=None)
                st.markdown('<div class="small">At-risk counts shown below the x-axis.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.info(f"At-risk table unavailable: {e}")
        st.pyplot(fig, use_container_width=True)

        # Stats
        with st.expander("📊 Statistics"):
            try:
                res = logrank_test(mA["km_time"], mB["km_time"], event_observed_A=mA["km_event"], event_observed_B=mB["km_event"])
                st.write(f"Log-rank test: p = **{res.p_value:.4g}**")
            except Exception as e:
                st.info(f"Log-rank failed: {e}")
            try:
                tmp = pd.concat([mA.assign(group=0), mB.assign(group=1)], axis=0, ignore_index=True)
                cph = CoxPHFitter()
                cph.fit(tmp[["km_time","km_event","group"]], duration_col="km_time", event_col="km_event")
                import numpy as np
                hr = float(np.exp(cph.params_["group"]))
                ci = cph.confidence_intervals_.loc["group"].values
                st.write(f"Cox PH HR (B vs A): **{hr:.3g}**  [{np.exp(ci[0]):.3g}, {np.exp(ci[1]):.3g}]")
            except Exception as e:
                st.info(f"Cox PH failed: {e}")

        # Export
        with st.expander("📥 Export"):
            out = pd.concat([mA.assign(__cohort=A["label"]), mB.assign(__cohort=B["label"])], ignore_index=True)
            cols = ["__cohort","patient_id","km_time","km_event","km_time_label","diagnosis_date","surgery_date","first_treatment_date","progression_date","death_date","last_followup_date","next_treatment_date","cancer_type","stage_group"]
            csv = out[[c for c in cols if c in out.columns]].to_csv(index=False).encode("utf-8")
            st.download_button("Download KM analysis dataset (CSV)", data=csv, file_name="km_onc_analysis.csv", mime="text/csv")

def main():
    st.markdown('<div class="header-bar"><div class="header-title">QS Oncology Stats (POC v5.1)</div><div class="header-sub">Flexible rule builder • Tests+Patients • OS/TTP/PFS/TTT/TTNT</div></div>', unsafe_allow_html=True)
    data_tab()
    cohorts_tab()
    endpoint_tab()
    km_stats_tab()

if __name__ == "__main__":
    main()
