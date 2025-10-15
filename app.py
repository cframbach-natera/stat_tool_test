
# app.py
# QS Oncology Stats – Cohort Builder v5.4 (no Summary tab)

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

st.set_page_config(page_title="QS Oncology Stats – Cohort Builder", page_icon="🧬", layout="wide")

PALETTE = {"blue":"#60A4BF","green":"#7EB86E","blue100":"#CEDFFB","blue200":"#A9CCF5","blue500":"#329ADB","blue900":"#1B5F7F","gray":"#8D97A1"}
mpl.rcParams.update({
    "axes.edgecolor": PALETTE["blue900"], "axes.labelcolor": PALETTE["blue900"],
    "xtick.color": PALETTE["blue900"], "ytick.color": PALETTE["blue900"],
    "axes.grid": True, "grid.alpha": 0.22, "grid.color": PALETTE["gray"],
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": True,
    "legend.facecolor": "white", "legend.edgecolor": PALETTE["blue900"],
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 12.0, "axes.titleweight": "semibold",
})

def render_css():
    st.markdown("""
    <style>
    :root { --qs-blue:#60A4BF; --qs-blue500:#329ADB; --qs-blue900:#1B5F7F; --qs-green:#7EB86E;
            --qs-gray-50:#F9FAFB; --qs-gray-100:#F2F4F7; --qs-gray-200:#E5E7EB; --qs-gray-300:#D0D5DD;
            --qs-gray-500:#667085; --qs-gray-700:#344054; --qs-ink:#101828;
            --primary-color:#329ADB; --secondary-background-color:#F2F4F7; --text-color:#101828;
            --qs-top-offset: 64px; }
    .block-container{padding-top:1rem;padding-bottom:2rem;}
    .stTabs [role="tablist"]{position:sticky;top:var(--qs-top-offset);z-index:1000;background:#fff;padding:6px 0 0;margin:0 0 8px;border-bottom:1px solid var(--qs-gray-200);}
    .stTabs [role="tab"]{color:var(--qs-ink)!important;opacity:1!important;font-weight:600!important;background:transparent!important;border-bottom:2px solid transparent!important;}
    .stTabs [role="tab"][aria-selected="true"]{color:var(--qs-blue900)!important;border-bottom-color:var(--qs-blue500)!important;}
    .header-bar{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:12px;background:var(--qs-gray-50);border:1px solid var(--qs-gray-200);margin:6px 0 12px;}
    .header-title{font-weight:800;color:var(--qs-ink);font-size:1.05rem;}
    .header-sub{color:var(--qs-gray-700);font-size:.9rem;font-weight:500;}
    .panel{padding:12px 14px;border-radius:14px;border:1px solid var(--qs-gray-200);background:#fff;box-shadow:0 1px 6px rgba(0,0,0,.04);}
    .kpi-card{border-radius:14px;padding:12px 16px;background:#fff;border:1px solid var(--qs-gray-200);box-shadow:0 1px 10px rgba(0,0,0,.04);}
    .kpi-title{font-weight:600;color:var(--qs-ink);} .kpi-value{font-size:1.15rem;font-weight:800;color:var(--qs-blue900);} .small{font-size:.85rem;color:var(--qs-gray-500);}
    .stButton>button{background-color:var(--qs-blue500)!important;border:1px solid var(--qs-blue900)!important;color:#fff!important;font-weight:700!important;border-radius:10px!important;}
    .btn-green .stButton>button{background-color:var(--qs-green)!important;border:1px solid #5f9e53!important;color:#0b2610!important;}
    .stMultiSelect [data-baseweb="tag"],.stMultiSelect span[data-baseweb="tag"],.stMultiSelect div[data-baseweb="tag"]{background:var(--qs-gray-100)!important;color:var(--qs-ink)!important;border-radius:10px!important;border:1px solid var(--qs-gray-300)!important;box-shadow:none!important;}
    .stMultiSelect [data-baseweb="tag"] svg{color:var(--qs-gray-500)!important;}
    .stSlider [data-baseweb="slider"]>div>div{background:var(--qs-blue500)!important;}
    .stSlider [role="slider"]{background:var(--qs-blue500)!important;border-color:var(--qs-blue500)!important;}
    input[type="radio"]:checked{accent-color:var(--qs-blue500)!important;}
    input[type="checkbox"]:checked{accent-color:var(--qs-green)!important;}
    .condition-card{border:1px dashed var(--qs-gray-300);border-radius:12px;padding:10px;margin-bottom:8px;background:#fff;}
    header, .stApp header { z-index: 1001; }
    </style>
    """, unsafe_allow_html=True)

def make_tabs():
    return st.tabs(["📁 Data", "👥 Cohorts", "🎯 Endpoint & Model", "📈 KM & Stats"])

PATIENT_ANCHORS = ["diagnosis_date","surgery_date","first_treatment_date","progression_date","next_treatment_date","death_date","last_followup_date"]

def load_patients_sample() -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv("patients_sample.csv", parse_dates=PATIENT_ANCHORS)
    except Exception:
        return None

def load_tests_sample() -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv("tests_sample.csv", parse_dates=["test_date"])
    except Exception:
        return None

def render_quick_filters(prefix: str, dfp: pd.DataFrame):
    st.subheader(f"Cohort {prefix} – quick filters")
    filters = {}
    label_key = f"{prefix}_label"
    label = st.text_input("Cohort label", value=st.session_state.get(label_key, f"Cohort {prefix}"), key=label_key)
    c1,c2,c3 = st.columns(3)
    if "age_at_diag" in dfp.columns:
        a_min,a_max = int(np.nanmin(dfp["age_at_diag"])), int(np.nanmax(dfp["age_at_diag"]))
        key_rng=f"{prefix}_age_range"; default_rng=st.session_state.get(key_rng,(a_min,a_max))
        key_mode=f"{prefix}_age_mode"; default_mode=st.session_state.get(key_mode,"Include")
        with c1:
            rng = st.slider("Age at diagnosis", min_value=a_min, max_value=a_max, value=(default_rng[0],default_rng[1]), key=key_rng)
            mode = st.radio("Filter mode", ["Include","Exclude"], horizontal=True, key=key_mode, index=0 if default_mode=="Include" else 1)
        filters["age_at_diag"]={"type":"range","value":rng,"include":(mode=="Include")}
    def cat_filter(col,title,container):
        if col not in dfp.columns: return
        opts = sorted([x for x in dfp[col].dropna().unique().tolist()], key=lambda x: str(x))
        key_vals=f"{prefix}_{col}_vals"; key_mode=f"{prefix}_{col}_mode"
        default_vals=st.session_state.get(key_vals,opts); default_mode=st.session_state.get(key_mode,"Include")
        with container:
            sel = st.multiselect(title, options=opts, default=default_vals, key=key_vals)
            mode = st.radio("Mode", ["Include","Exclude"], horizontal=True, key=key_mode, index=0 if default_mode=="Include" else 1)
        if sel or mode=="Exclude":
            filters[col]={"type":"set","values":set(sel),"include":(mode=="Include")}
    cat_filter("sex","Biological sex",c1); cat_filter("race","Race",c2); cat_filter("ethnicity","Ethnicity",c2); cat_filter("stage_group","Stage group",c3); cat_filter("cancer_type","Cancer type",c3)
    return label, filters

RESULT_VALUES = ["positive","negative","indeterminate"]
ANCHOR_COMPARE_OPS = ["<","<=",">",">=","==","!="]
MISSING_POLICY = ["fail","pass"]

def init_rules(prefix:str):
    st.session_state.setdefault(f"{prefix}_rules", [])
    st.session_state.setdefault(f"{prefix}_rules_logic","ALL")

def add_condition(prefix:str):
    default_cond={"family":"tests","type":"Count","start_anchor":"surgery_date","start_offset":0,"end_mode":"anchor","end_anchor":"first_treatment_date","end_offset":90,"include_start":True,"include_end":True,"results":["positive","negative"],"require_result":"positive","include_indeterminate":False,"count_op":"at least","count_n":1,"consecutive_k":2,"proportion_pct":50,"min_tests":1,"facet_filters":{},"anchor_type":"AnchorExists","anchor_exists_target":"first_treatment_date","anchor_exists_should":True,"anchor_left":"first_treatment_date","anchor_right":"progression_date","anchor_op":"<","missing_left_policy":"fail","missing_right_policy":"fail"}
    st.session_state[f"{prefix}_rules"].append(default_cond)

def render_condition(prefix:str, idx:int, tests_df: Optional[pd.DataFrame]):
    cond = st.session_state[f"{prefix}_rules"][idx]
    st.markdown('<div class="condition-card">', unsafe_allow_html=True)
    fam = st.radio("Condition family", ["tests","anchor"], horizontal=True, index=0 if cond.get("family","tests")=="tests" else 1, key=f"{prefix}_fam_{idx}")
    cond["family"]=fam
    if fam=="tests":
        c1,c2,c3 = st.columns([1.2,1,1])
        with c1:
            cond["type"]=st.selectbox("Condition type",["Count","Exists","None","FirstResult","LastResult","Consecutive","Proportion"], index=["Count","Exists","None","FirstResult","LastResult","Consecutive","Proportion"].index(cond["type"]), key=f"{prefix}_type_{idx}")
        with c2:
            cond["start_anchor"]=st.selectbox("Start anchor", PATIENT_ANCHORS, index=PATIENT_ANCHORS.index(cond["start_anchor"]), key=f"{prefix}_sa_{idx}")
            cond["start_offset"]=st.number_input("Start offset (days)", value=int(cond["start_offset"]), step=1, key=f"{prefix}_so_{idx}")
        with c3:
            cond["end_mode"]=st.radio("End by",["anchor","offset"], index=0 if cond["end_mode"]=="anchor" else 1, horizontal=True, key=f"{prefix}_em_{idx}")
            if cond["end_mode"]=="anchor":
                cond["end_anchor"]=st.selectbox("End anchor", PATIENT_ANCHORS, index=PATIENT_ANCHORS.index(cond["end_anchor"]), key=f"{prefix}_ea_{idx}")
            else:
                cond["end_offset"]=st.number_input("End offset (days)", value=int(cond["end_offset"]), step=1, key=f"{prefix}_eo_{idx}")
        c4,c5=st.columns(2)
        with c4: cond["include_start"]=st.checkbox("Include start day", value=bool(cond["include_start"]), key=f"{prefix}_is_{idx}")
        with c5: cond["include_end"]=st.checkbox("Include end day", value=bool(cond["include_end"]), key=f"{prefix}_ie_{idx}")
        if cond["type"] in ["FirstResult","LastResult"]:
            cond["require_result"]=st.selectbox("Required result", RESULT_VALUES, index=RESULT_VALUES.index(cond["require_result"]), key=f"{prefix}_req_{idx}")
            cond["include_indeterminate"]=st.checkbox("Consider indeterminate in ordering", value=bool(cond["include_indeterminate"]), key=f"{prefix}_ind_{idx}")
        elif cond["type"] in ["Count","Exists","None","Consecutive","Proportion"]:
            cond["results"]=st.multiselect("Test results to include", RESULT_VALUES, default=cond["results"], key=f"{prefix}_res_{idx}")
            cond["include_indeterminate"]=st.checkbox("Include indeterminate in pool", value=bool(cond["include_indeterminate"]), key=f"{prefix}_ind2_{idx}")
            if cond["type"]=="Count":
                cond["count_op"]=st.selectbox("Comparator",["at least","at most","exactly"], index=["at least","at most","exactly"].index(cond["count_op"]), key=f"{prefix}_cop_{idx}")
                cond["count_n"]=st.number_input("Count N", value=int(cond["count_n"]), min_value=0, step=1, key=f"{prefix}_cn_{idx}")
            if cond["type"]=="Consecutive":
                cond["consecutive_k"]=st.number_input("Consecutive K", value=int(cond["consecutive_k"]), min_value=1, step=1, key=f"{prefix}_ck_{idx}")
            if cond["type"]=="Proportion":
                cond["proportion_pct"]=st.number_input("Proportion threshold (%)", value=int(cond["proportion_pct"]), min_value=0, max_value=100, step=1, key=f"{prefix}_pp_{idx}")
                cond["min_tests"]=st.number_input("Minimum tests in window", value=int(cond["min_tests"]), min_value=0, step=1, key=f"{prefix}_mt_{idx}")
        if tests_df is not None:
            with st.expander("Facet filters (optional)"):
                for fc in [c for c in ["assay_version","specimen_type","lab_site"] if c in tests_df.columns]:
                    opts=sorted([x for x in tests_df[fc].dropna().unique().tolist()], key=lambda x: str(x))
                    sel=st.multiselect(fc, options=opts, default=cond["facet_filters"].get(fc,opts), key=f"{prefix}_facet_{idx}_{fc}")
                    if len(sel)!=len(opts): cond["facet_filters"][fc]=sel
                    elif fc in cond["facet_filters"]: del cond["facet_filters"][fc]
    else:
        cond["anchor_type"]=st.selectbox("Anchor condition",["AnchorExists","AnchorOrder"], index=["AnchorExists","AnchorOrder"].index(cond.get("anchor_type","AnchorExists")), key=f"{prefix}_anchtype_{idx}")
        if cond["anchor_type"]=="AnchorExists":
            c1,c2=st.columns(2)
            with c1:
                cond["anchor_exists_target"]=st.selectbox("Anchor", PATIENT_ANCHORS, index=PATIENT_ANCHORS.index(cond.get("anchor_exists_target","first_treatment_date")), key=f"{prefix}_axt_{idx}")
            with c2:
                cond["anchor_exists_should"]=st.selectbox("Should exist?", [True,False], index=0 if cond.get("anchor_exists_should",True) else 1, key=f"{prefix}_axs_{idx}")
                st.caption("True = must exist; False = must NOT exist.")
        else:
            c1,c2,c3=st.columns([1,0.5,1])
            with c1: cond["anchor_left"]=st.selectbox("Left anchor", PATIENT_ANCHORS, index=PATIENT_ANCHORS.index(cond.get("anchor_left","first_treatment_date")), key=f"{prefix}_al_{idx}")
            with c2: cond["anchor_op"]=st.selectbox("Operator", ANCHOR_COMPARE_OPS, index=ANCHOR_COMPARE_OPS.index(cond.get("anchor_op","<")), key=f"{prefix}_aop_{idx}")
            with c3: cond["anchor_right"]=st.selectbox("Right anchor", PATIENT_ANCHORS, index=PATIENT_ANCHORS.index(cond.get("anchor_right","progression_date")), key=f"{prefix}_ar_{idx}")
            d1,d2=st.columns(2)
            with d1: cond["missing_left_policy"]=st.selectbox("If left missing", MISSING_POLICY, index=MISSING_POLICY.index(cond.get("missing_left_policy","fail")), key=f"{prefix}_mlp_{idx}")
            with d2: cond["missing_right_policy"]=st.selectbox("If right missing", MISSING_POLICY, index=MISSING_POLICY.index(cond.get("missing_right_policy","fail")), key=f"{prefix}_mrp_{idx}")
    if st.button("Delete condition", key=f"{prefix}_del_{idx}"):
        st.session_state[f"{prefix}_rules"].pop(idx); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def window_for_condition(patients: pd.DataFrame, cond: Dict) -> pd.DataFrame:
    start = pd.to_datetime(patients[cond["start_anchor"]], errors="coerce") + pd.to_timedelta(int(cond["start_offset"]), unit="D")
    if cond["end_mode"]=="anchor":
        end = pd.to_datetime(patients[cond["end_anchor"]], errors="coerce")
    else:
        end = start + pd.to_timedelta(int(cond["end_offset"]), unit="D")
    return pd.DataFrame({"patient_id": patients["patient_id"], "win_start": start, "win_end": end})

def filter_tests_in_window(tests: pd.DataFrame, win: pd.DataFrame, cond: Dict) -> pd.DataFrame:
    d = tests.merge(win, on="patient_id", how="left")
    left_ok = d["test_date"] >= d["win_start"] if cond["include_start"] else d["test_date"] > d["win_start"]
    right_ok = (d["win_end"].isna()) | (d["test_date"] <= d["win_end"]) if cond["include_end"] else (d["win_end"].isna()) | (d["test_date"] < d["win_end"])
    d = d[left_ok & right_ok].copy()
    if cond["type"] in ["Count","Exists","None","Consecutive","Proportion"]:
        if cond["results"]:
            res_set=set(cond["results"]); 
            if cond.get("include_indeterminate", False): res_set.add("indeterminate")
            d = d[d["result"].isin(res_set)]
        elif not cond.get("include_indeterminate", False):
            d = d[d["result"].isin(["positive","negative"])]
    for fc, vals in cond.get("facet_filters", {}).items():
        if fc in d.columns: d = d[d[fc].isin(vals)]
    return d

def eval_anchor_exists(patients: pd.DataFrame, target: str, should_exist: bool) -> pd.Series:
    series = pd.to_datetime(patients[target], errors="coerce"); exists = series.notna()
    return exists if should_exist else ~exists

def eval_anchor_order(patients: pd.DataFrame, left: str, op: str, right: str, miss_left: str, miss_right: str) -> pd.Series:
    L = pd.to_datetime(patients[left], errors="coerce"); R = pd.to_datetime(patients[right], errors="coerce")
    l_missing=L.isna(); r_missing=R.isna(); base=pd.Series(False, index=patients.index); both=~(l_missing|r_missing)
    if op=="<": base[both]=(L[both] < R[both])
    elif op=="<=": base[both]=(L[both] <= R[both])
    elif op==">": base[both]=(L[both] > R[both])
    elif op==">=": base[both]=(L[both] >= R[both])
    elif op=="==": base[both]=(L[both] == R[both])
    else: base[both]=(L[both] != R[both])
    if miss_left=="pass": base[l_missing & ~r_missing]=True
    if miss_right=="pass": base[~l_missing & r_missing]=True
    return base.fillna(False)

def eval_condition(patients: pd.DataFrame, tests: pd.DataFrame, cond: Dict) -> pd.Series:
    if cond.get("family","tests")=="anchor":
        if cond.get("anchor_type","AnchorExists")=="AnchorExists":
            return eval_anchor_exists(patients, cond["anchor_exists_target"], bool(cond["anchor_exists_should"]))
        else:
            return eval_anchor_order(patients, cond["anchor_left"], cond["anchor_op"], cond["anchor_right"], cond.get("missing_left_policy","fail"), cond.get("missing_right_policy","fail"))
    win = window_for_condition(patients, cond)
    d = filter_tests_in_window(tests, win, cond)
    grouped = d.groupby("patient_id")
    if cond["type"]=="Exists":
        has_any = grouped.size().rename("cnt")
        return patients["patient_id"].map(has_any.gt(0)).fillna(False)
    if cond["type"]=="None":
        has_any = grouped.size().rename("cnt")
        return ~patients["patient_id"].map(has_any.gt(0)).fillna(False)
    if cond["type"]=="Count":
        counts = grouped.size().rename("cnt"); op=cond["count_op"]; n=int(cond["count_n"])
        if op=="at least": ok=counts.ge(n)
        elif op=="at most": ok=counts.le(n)
        else: ok=counts.eq(n)
        return patients["patient_id"].map(ok).fillna(False)
    if cond["type"] in ["FirstResult","LastResult"]:
        dd = tests.merge(win, on="patient_id", how="left")
        left_ok = dd["test_date"] >= dd["win_start"] if cond["include_start"] else dd["test_date"] > dd["win_start"]
        right_ok = (dd["win_end"].isna()) | (dd["test_date"] <= dd["win_end"]) if cond["include_end"] else (dd["win_end"].isna()) | (dd["test_date"] < dd["win_end"])
        dd = dd[left_ok & right_ok].copy()
        if not cond["include_indeterminate"]: dd = dd[dd["result"]!="indeterminate"]
        order = dd.sort_values(["patient_id","test_date"])
        pick = order.groupby("patient_id").first()["result"] if cond["type"]=="FirstResult" else order.groupby("patient_id").last()["result"]
        ok = pick.eq(cond["require_result"])
        return patients["patient_id"].map(ok).fillna(False)
    if cond["type"]=="Consecutive":
        k=int(cond["consecutive_k"])
        if k<=1: has_any=grouped.size().rename("cnt").gt(0); return patients["patient_id"].map(has_any).fillna(False)
        ok_ids=set(); sel=set(cond.get("results", [])); 
        if cond.get("include_indeterminate", False): sel.add("indeterminate")
        for pid, g in grouped:
            g=g.sort_values("test_date"); seq=g["result"].tolist(); run=0
            for r in seq:
                if r in sel:
                    run+=1
                    if run>=k: ok_ids.add(pid); break
                else: run=0
        return patients["patient_id"].isin(ok_ids)
    if cond["type"]=="Proportion":
        pct=float(cond["proportion_pct"]); min_tests=int(cond.get("min_tests",1))
        cnts=grouped.size().rename("n_total"); sel=set(cond.get("results", []))
        if cond.get("include_indeterminate", False): sel.add("indeterminate")
        in_sel=d[d["result"].isin(sel)]; cnt_in=in_sel.groupby("patient_id").size().rename("n_in")
        joined=pd.concat([cnts,cnt_in],axis=1).fillna(0); prop=(joined["n_in"]/joined["n_total"]).where(joined["n_total"]>=min_tests,0.0)
        ok=prop.ge(pct/100.0); return patients["patient_id"].map(ok).fillna(False)
    return pd.Series(False, index=patients.index)

def apply_quick_filters(patients: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    if not filters: return patients.copy()
    d=patients.copy(); mask=pd.Series(True, index=d.index)
    for col, spec in filters.items():
        ftype=spec.get("type"); include=spec.get("include",True)
        if ftype=="range":
            lo,hi=spec["value"]; inside=d[col].astype(float).between(lo,hi)
            mask &= (inside if include else ~inside)
        elif ftype in ("set","in"):
            vals=spec.get("values", set()); in_set=d[col].isin(vals)
            mask &= (in_set if include else ~in_set)
    return d[mask].copy()

def data_tab(tab):
    with tab:
        st.markdown('<div class="header-bar"><div class="header-title">Data</div><div class="header-sub">Upload patients & tests or use the bundled samples.</div></div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            use_pat_sample = st.toggle("Use patients sample", value=True, key="use_pat_sample")
            pat_file = st.file_uploader("Upload patients.csv", type=["csv"], key="pat_upl")
            dfp=None
            if use_pat_sample and pat_file is None:
                try:
                    dfp = pd.read_csv("patients_sample.csv", parse_dates=["diagnosis_date","surgery_date","first_treatment_date","progression_date","next_treatment_date","death_date","last_followup_date"])
                    st.download_button("Download patients sample", data=dfp.to_csv(index=False).encode("utf-8"), file_name="patients_sample.csv", mime="text/csv")
                except Exception: dfp=None
            elif pat_file is not None:
                dfp = pd.read_csv(pat_file, parse_dates=["diagnosis_date","surgery_date","first_treatment_date","progression_date","next_treatment_date","death_date","last_followup_date"], infer_datetime_format=True)
        with c2:
            use_tests_sample = st.toggle("Use tests sample", value=True, key="use_tests_sample")
            tst_file = st.file_uploader("Upload tests.csv", type=["csv"], key="tst_upl")
            dft=None
            if use_tests_sample and tst_file is None:
                try:
                    dft = pd.read_csv("tests_sample.csv", parse_dates=["test_date"])
                    st.download_button("Download tests sample", data=dft.to_csv(index=False).encode("utf-8"), file_name="tests_sample.csv", mime="text/csv")
                except Exception: dft=None
            elif tst_file is not None:
                dft = pd.read_csv(tst_file, parse_dates=["test_date"], infer_datetime_format=True)
        st.session_state["patients_df"]=dfp; st.session_state["tests_df"]=dft
        if dfp is not None and dft is not None:
            st.success(f"Loaded {len(dfp):,} patients and {len(dft):,} tests.")
            with st.expander("Preview patients (top 20)"): st.dataframe(dfp.head(20))
            with st.expander("Preview tests (top 20)"): st.dataframe(dft.head(20))
        else:
            st.info("Load both patients and tests to continue.")

def build_cohort_ui(prefix:str):
    patients = st.session_state.get("patients_df"); tests = st.session_state.get("tests_df")
    if patients is None or tests is None:
        st.info("Load data in the **Data** tab first."); return None
    init_rules(prefix)
    st.markdown("### Conditions")
    st.session_state[f"{prefix}_rules_logic"]=st.radio("Combine with",["ALL","ANY"], horizontal=True, key=f"{prefix}_agg")
    st.button("Add condition", on_click=add_condition, args=(prefix,), key=f"{prefix}_add")
    for i in range(len(st.session_state[f"{prefix}_rules"])):
        render_condition(prefix, i, tests)
    st.markdown("### Quick demographic/clinical filters")
    label, qfilters = render_quick_filters(prefix, patients)
    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown('<div class="btn-green">', unsafe_allow_html=True)
        if st.button(f"Build {label}", key=f"{prefix}_build"):
            st.session_state[f"{prefix}_built"]=True
            st.session_state[f"{prefix}_label_built"]=label
            st.session_state[f"{prefix}_qf_built"]=qfilters
            st.session_state[f"{prefix}_rules_built"]=st.session_state[f"{prefix}_rules"]
            st.session_state[f"{prefix}_logic_built"]=st.session_state[f"{prefix}_rules_logic"]
            st.success(f"Cohort {prefix} built.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if st.button(f"Reset {prefix}", key=f"{prefix}_reset"):
            for k in [f"{prefix}_built", f"{prefix}_label_built", f"{prefix}_qf_built", f"{prefix}_rules_built", f"{prefix}_logic_built", f"{prefix}_rules"]:
                if k in st.session_state: del st.session_state[k]
            init_rules(prefix); st.info(f"Cohort {prefix} reset.")
    built = st.session_state.get(f"{prefix}_built", False)
    label_built = st.session_state.get(f"{prefix}_label_built", label)
    qf_built = st.session_state.get(f"{prefix}_qf_built", qfilters)
    rules_built = st.session_state.get(f"{prefix}_rules_built", st.session_state.get(f"{prefix}_rules", []))
    logic_built = st.session_state.get(f"{prefix}_logic_built", st.session_state.get(f"{prefix}_rules_logic","ALL"))
    return {"label":label_built,"qf":qf_built,"rules":rules_built,"logic":logic_built,"built":built}

def cohorts_tab(tab):
    with tab:
        st.markdown('<div class="header-bar"><div class="header-title">Cohorts</div><div class="header-sub">Build all relationships via rules. No flags.</div></div>', unsafe_allow_html=True)
        patients = st.session_state.get("patients_df"); tests = st.session_state.get("tests_df")
        if patients is None or tests is None:
            st.info("Load data in the **Data** tab first."); return
        left,right = st.columns(2)
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True); A = build_cohort_ui("A"); st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True); B = build_cohort_ui("B"); st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Reset both cohorts"):
            for p in ["A","B"]:
                for k in [f"{p}_built", f"{p}_label_built", f"{p}_qf_built", f"{p}_rules_built", f"{p}_logic_built", f"{p}_rules"]:
                    if k in st.session_state: del st.session_state[k]
            init_rules("A"); init_rules("B"); st.info("Both cohorts reset.")
        st.session_state["cohortA"]=A; st.session_state["cohortB"]=B

def endpoint_tab(tab):
    with tab:
        st.markdown('<div class="header-bar"><div class="header-title">Endpoint & Model</div><div class="header-sub">Choose endpoint and baseline.</div></div>', unsafe_allow_html=True)
        patients = st.session_state.get("patients_df"); tests = st.session_state.get("tests_df")
        if patients is None or tests is None:
            st.info("Load data in the **Data** tab first."); return
        ev_type = st.radio("Endpoint", ["Overall Survival (OS): death","Time to Progression (TTP): progression only","Progression-Free Survival (PFS): progression or death","Time to Treatment (TTT): first treatment","Time to Next Treatment (TTNT): next treatment"], index=0, key="ev_type")
        baseline = st.radio("Baseline (Day 0)", ["Diagnosis date","Positive test date (first)","Surgery date","First treatment date"], index=2, key="baseline_choice")
        unit = st.radio("Time unit", ["days","months","years"], index=1, key="time_unit")
        st.session_state["endpoint_cfg"]={"type":ev_type,"baseline":baseline,"unit":unit}; st.success("Endpoint settings saved.")

def choose_baseline_series(patients: pd.DataFrame, cfg: Dict, tests: Optional[pd.DataFrame]) -> pd.Series:
    if cfg["baseline"].startswith("Diagnosis"): return pd.to_datetime(patients["diagnosis_date"], errors="coerce")
    if cfg["baseline"].startswith("Surgery"): return pd.to_datetime(patients["surgery_date"], errors="coerce")
    if cfg["baseline"].startswith("First treatment"): return pd.to_datetime(patients["first_treatment_date"], errors="coerce")
    if tests is None or tests.empty: return pd.Series(pd.NaT, index=patients.index)
    first_pos = tests[tests["result"]=="positive"].sort_values(["patient_id","test_date"]).groupby("patient_id").first()["test_date"]
    return patients["patient_id"].map(first_pos)

def compute_km_dataset(patients: pd.DataFrame, tests: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    d=patients.copy(); base=choose_baseline_series(d, cfg, tests)
    death=pd.to_datetime(d["death_date"], errors="coerce"); prog=pd.to_datetime(d["progression_date"], errors="coerce")
    next_tx=pd.to_datetime(d["next_treatment_date"], errors="coerce"); first_tx=pd.to_datetime(d["first_treatment_date"], errors="coerce")
    censor=pd.to_datetime(d["last_followup_date"], errors="coerce")
    if cfg["type"].startswith("Overall Survival"):
        event=(~death.isna()) & (~base.isna()) & (death>base); t_end=np.where(event, death, censor)
    elif cfg["type"].startswith("Time to Progression"):
        event=(~prog.isna()) & (~base.isna()) & (prog>base); t_end=np.where(event, prog, censor)
    elif cfg["type"].startswith("Progression-Free Survival"):
        cand=pd.concat([prog,death],axis=1).min(axis=1); event=(~cand.isna()) & (~base.isna()) & (cand>base); t_end=np.where(event, cand, censor)
    elif cfg["type"].startswith("Time to Treatment (TTT)"):
        event=(~first_tx.isna()) & (~base.isna()) & (first_tx>base); t_end=np.where(event, first_tx, censor)
    else:
        event=(~next_tx.isna()) & (~base.isna()) & (next_tx>base); t_end=np.where(event, next_tx, censor)
    t_end=pd.to_datetime(t_end); time_days=(t_end-base).dt.days.astype("float")
    d["km_time_days"]=time_days; d["km_event"]=event.astype(int)
    unit=cfg["unit"]
    if unit=="months": d["km_time"]=d["km_time_days"]/30.4375; d["km_time_label"]="Months"
    elif unit=="years": d["km_time"]=d["km_time_days"]/365.25; d["km_time_label"]="Years"
    else: d["km_time"]=d["km_time_days"]; d["km_time_label"]="Days"
    valid=(~d["km_time"].isna()) & np.isfinite(d["km_time"]) & (d["km_time"]>=0)
    return d[valid].copy()

def km_stats_tab(tab):
    with tab:
        st.markdown('<div class="header-bar"><div class="header-title">KM & Stats</div><div class="header-sub">Survival curves, at-risk table, and tests.</div></div>', unsafe_allow_html=True)
        patients=st.session_state.get("patients_df"); tests=st.session_state.get("tests_df")
        if patients is None or tests is None: st.info("Load data in the **Data** tab first."); return
        A=st.session_state.get("cohortA"); B=st.session_state.get("cohortB")
        if not A or not B or not (A["built"] and B["built"]): st.info("Build **both** cohorts in the Cohorts tab."); return
        cfg=st.session_state.get("endpoint_cfg"); if not cfg: st.info("Set an endpoint in the **Endpoint & Model** tab."); return
        dA_q=apply_quick_filters(patients, A["qf"]); dB_q=apply_quick_filters(patients, B["qf"])
        def apply_with_snapshot(df, snap):
            rules=snap["rules"]; logic=snap["logic"]
            if not rules: return df
            masks=[eval_condition(df, st.session_state["tests_df"], c) for c in rules]
            if logic=="ALL":
                m=masks[0]
                for mm in masks[1:]: m &= mm
            else:
                m=masks[0]
                for mm in masks[1:]: m |= mm
            return df[m].copy()
        dA=apply_with_snapshot(dA_q, A); dB=apply_with_snapshot(dB_q, B)
        if dA.empty or dB.empty: st.warning("One cohort is empty after filters/rules."); return
        mA=compute_km_dataset(dA, tests, cfg); mB=compute_km_dataset(dB, tests, cfg)
        kcols=st.columns(2)
        for i,(lab,d) in enumerate([(A["label"],mA),(B["label"],mB)]):
            try:
                km=KaplanMeierFitter().fit(d["km_time"], d["km_event"]); med=float(km.median_survival_time_) if km.median_survival_time_ is not None else np.nan
            except Exception: med=np.nan
            med_txt="NA" if (not np.isfinite(med)) else f"{med:.2f}"; rate=f"{(d['km_event'].sum()/len(d)*100):.0f}%" if len(d) else "—"
            html=f'<div class="kpi-card"><div class="kpi-title">{lab}</div><div class="kpi-value">Median: {med_txt}</div><div class="small">N={len(d)} • Events={int(d["km_event"].sum())} ({rate})</div></div>'
            with kcols[i]: st.markdown(html, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9.6,5.8)); fitters=[]; colors=[PALETTE["blue500"], PALETTE["green"]]
        for (lab,d), color in zip([(A["label"],mA),(B["label"],mB)], colors):
            km=KaplanMeierFitter()
            try:
                km.fit(durations=d["km_time"], event_observed=d["km_event"], label=str(lab))
                km.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.2); fitters.append(km)
            except Exception as e: st.warning(f"KM error for {lab}: {e}")
        ax.set_xlabel(f"Time ({mA['km_time_label'].iloc[0] if len(mA) else 'time'})", color=PALETTE["blue900"]); ax.set_ylabel("Survival probability", color=PALETTE["blue900"])
        leg=ax.legend(framealpha=0.95, title=None); [t.set_color(PALETTE["blue900"]) for t in leg.get_texts()]
        try:
            if len(fitters)>=1: add_at_risk_counts(*fitters, ax=ax, xticks=None); st.markdown('<div class="small">At-risk counts shown below the x-axis.</div>', unsafe_allow_html=True)
        except Exception as e: st.info(f"At-risk table unavailable: {e}")
        st.pyplot(fig, use_container_width=True)
        with st.expander("📊 Statistics"):
            try:
                res=logrank_test(mA["km_time"], mB["km_time"], event_observed_A=mA["km_event"], event_observed_B=mB["km_event"])
                st.write(f"Log-rank test: p = **{res.p_value:.4g}**")
            except Exception as e: st.info(f"Log-rank failed: {e}")
            try:
                tmp=pd.concat([mA.assign(group=0), mB.assign(group=1)], ignore_index=True); cph=CoxPHFitter()
                cph.fit(tmp[["km_time","km_event","group"]], duration_col="km_time", event_col="km_event")
                hr=float(np.exp(cph.params_["group"])); ci_low, ci_high = cph.confidence_intervals_.loc["group"].values
                st.write(f"Cox PH HR (B vs A): **{hr:.3g}**  [{np.exp(ci_low):.3g}, {np.exp(ci_high):.3g}]")
            except Exception as e: st.info(f"Cox PH failed: {e}")

def make_tabs():
    return st.tabs(["📁 Data", "👥 Cohorts", "🎯 Endpoint & Model", "📈 KM & Stats"])

def main():
    render_css()
    tabs = make_tabs()
    data_tab(tabs[0])
    cohorts_tab(tabs[1])
    endpoint_tab(tabs[2])
    km_stats_tab(tabs[3])

if __name__ == "__main__":
    main()
