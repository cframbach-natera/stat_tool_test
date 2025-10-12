
import os
import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Natera Oncology Demo", layout="wide")

PALETTE = {"blue_main": "#60A4BF", "green": "#7EB86E", "blue_100": "#CEDFFB", "blue_200": "#A9CCF5", "blue_500": "#329ADB", "teal_800": "#1B5F7F", "gray_500": "#8D97A1"}
ONCOPRINT_COLORS = {"AMP": "#E31A1C", "HOMDEL": "#1F78B4", "MISSENSE": "#008000", "TRUNC": "#000000", "INFRAME": "#708090", "SPLICE": "#FF8C00", "FUSION": "#6A3D9A"}

st.markdown("""
<style>
:root {
  --brand-primary: #329ADB;
  --brand-secondary: #1B5F7F;
  --brand-accent: #7EB86E;
  --bg: #ffffff; --subtle: #CEDFFB; --muted: #A9CCF5;
  --text: #0f172a; --text-muted: #8D97A1;
}
.block-container { padding-top: 0.6rem; }
.stTabs [data-baseweb="tab-list"] button[role="tab"] {
  border: 1px solid #e5e7eb; background:#fff; margin-right:8px; padding:8px 14px; border-radius:12px;
}
.stTabs [aria-selected="true"] { background: var(--brand-primary); color:#fff; border-color: var(--brand-primary); }
.card { border: 1px solid #e5e7eb; border-radius:14px; background:#fff; box-shadow:0 2px 14px rgba(2,6,23,.06); padding:14px 16px; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
SAMPLE = BASE / "sample_data"

def resolve(path_hint, default):
    p = Path(path_hint)
    if p.exists():
        return p
    # try relative to app
    p2 = BASE / path_hint
    if p2.exists():
        return p2
    # fall back to sample
    fallback = SAMPLE / default
    return fallback

@st.cache_data
def load_csv(path: str):
    p = resolve(path, Path(path).name)
    if not p.exists():
        st.warning(f"Couldn't find {path} — using sample {p}")
    return pd.read_csv(p)

tabs = st.tabs(["Time-to-event","OncoPrint","Treatments","Longitudinal","Response & Safety","QC"])

# Sidebar paths
with st.sidebar:
    st.header("Data")
    p_tte  = st.text_input("Time-to-event CSV", "sample_data/timetoevent.csv")
    p_mol  = st.text_input("Molecular CSV", "sample_data/genomics.csv")
    p_tx   = st.text_input("Treatments CSV", "sample_data/treatments.csv")
    p_long = st.text_input("Longitudinal CSV", "sample_data/longitudinal.csv")
    p_resp = st.text_input("Response visits CSV", "sample_data/response_visits.csv")
    p_ae   = st.text_input("Adverse events CSV", "sample_data/adverse_events.csv")
    p_qc   = st.text_input("QC CSV", "sample_data/qc.csv")

# ----- 1) KM
with tabs[0]:
    st.header("Kaplan–Meier")
    df = load_csv(p_tte)
    try:
        df["start_date"] = pd.to_datetime(df["start_date"]); df["event_date"] = pd.to_datetime(df["event_date"])
        df["time_days"] = (df["event_date"] - df["start_date"]).dt.days.clip(lower=0)
        strata_col = "strata" if "strata" in df.columns else None
        groups = ["All"] + (sorted(df[strata_col].dropna().unique().tolist()) if strata_col else [])
        pick = st.selectbox("Stratum", groups, index=0)
        sub = df if pick=="All" else df[df[strata_col]==pick]
        t = sub.sort_values("time_days")["time_days"].values
        e = sub.sort_values("time_days")["event_observed"].astype(int).values
        uniq = np.unique(t); at_risk=len(t); surv_t=[0]; surv_s=[1.0]
        for tt in uniq:
            d = int(e[t==tt].sum())
            if at_risk>0:
                surv_t.append(tt); surv_s.append(surv_s[-1]*(1-d/at_risk))
            at_risk -= int((t==tt).sum())
        fig = plt.figure(); plt.step(surv_t, surv_s, where="post"); plt.ylim(0,1.05)
        plt.xlabel("Days"); plt.ylabel("Survival probability")
        st.pyplot(fig, use_container_width=True)
    except Exception as ex:
        st.error(f"KM failed: {ex}")

# ----- 2) OncoPrint
with tabs[1]:
    st.header("OncoPrint")
    df = load_csv(p_mol)
    def token(row):
        vt=str(row.get("variant_type",""))
        vc=str(row.get("variant_classification","")).lower()
        cn=str(row.get("cnv_call","")).lower()
        if vt=="CNV":
            if cn.startswith("amp"): return "AMP"
            if "deep" in cn or "del" in cn: return "HOMDEL"
        if vt=="Fusion": return "FUSION"
        if "missense" in vc: return "MISSENSE"
        if "nonsense" in vc or "frameshift" in vc: return "TRUNC"
        if "in_frame" in vc: return "INFRAME"
        if "splice" in vc: return "SPLICE"
        return None
    try:
        df["event_token"]=df.apply(token, axis=1); df=df[~df["event_token"].isna()].copy()
        genes = sorted(df["gene"].unique().tolist())[:30]
        samples = sorted(df["sample_id"].unique().tolist())
        from collections import defaultdict
        grid = {s: {g: [] for g in genes} for s in samples}
        for _,r in df.iterrows():
            if r["gene"] in genes: grid[r["sample_id"]][r["gene"]].append(r["event_token"])
        fig, ax = plt.subplots(figsize=(max(6,len(genes)*0.35), max(5,len(samples)*0.12)))
        ax.set_xlim(0, len(genes)); ax.set_ylim(0, len(samples))
        ax.set_xticks(np.arange(len(genes))+0.5); ax.set_xticklabels(genes, rotation=90)
        ax.set_yticks(np.arange(len(samples))+0.5); ax.set_yticklabels(samples, fontsize=8)
        ax.invert_yaxis(); ax.set_facecolor("#f8fafc")
        def draw_cell(x,y,tokens):
            if not tokens: return
            ax.add_patch(plt.Rectangle((x,y),1,1,facecolor="#e5e7eb", edgecolor="#cbd5e1", linewidth=0.5))
            order=["AMP","HOMDEL","FUSION","TRUNC","MISSENSE","INFRAME","SPLICE"]
            toks=[t for t in order if t in tokens][:3]; n=len(toks)
            for i,t in enumerate(toks):
                color=ONCOPRINT_COLORS.get(t, "#111827"); h=1.0/n
                ax.add_patch(plt.Rectangle((x, y+i*h), 1, h, facecolor=color, edgecolor="#e5e7eb", linewidth=0.3))
        for yi,s in enumerate(samples):
            for xi,g in enumerate(genes):
                draw_cell(xi, yi, grid[s][g])
        ax.grid(False); ax.tick_params(length=0)
        st.pyplot(fig, use_container_width=True)
    except Exception as ex:
        st.error(f"OncoPrint failed: {ex}")

# ----- 3) Treatments
with tabs[2]:
    st.header("Treatment patterns")
    df = load_csv(p_tx)
    try:
        df["lot_start_date"]=pd.to_datetime(df["lot_start_date"]); df["lot_end_date"]=pd.to_datetime(df["lot_end_date"]); 
        pats = df["patient_id"].unique()[:12]; sdf=df[df["patient_id"].isin(pats)].copy()
        fig = plt.figure(figsize=(8,6)); y=0; yt=[]; yl=[]
        for pid in pats:
            sub=sdf[sdf["patient_id"]==pid].sort_values("line_number")
            base=sub["lot_start_date"].min()
            for _,r in sub.iterrows():
                x0=(r["lot_start_date"]-base).days; w=max(1,(r["lot_end_date"]-r["lot_start_date"]).days)
                plt.barh(y,w,left=x0); y+=0.4
            yt.append(y-0.2); yl.append(pid); y+=0.6
        plt.yticks(yt,yl); plt.xlabel("Days")
        st.pyplot(fig, use_container_width=True)
    except Exception as ex:
        st.error(f"Treatments plot failed: {ex}")

# ----- 4) Longitudinal
with tabs[3]:
    st.header("Longitudinal / Organ Health")
    df = load_csv(p_long)
    try:
        measure = st.selectbox("Measure", sorted(df["measure_name"].unique()))
        sub=df[df["measure_name"]==measure].copy(); sub["measure_date"]=pd.to_datetime(sub["measure_date"])
        agg=sub.groupby("measure_date")["measure_value"].agg(["mean","std"]).reset_index()
        fig=plt.figure(figsize=(8,4)); x=agg["measure_date"]; y=agg["mean"].values; sd=agg["std"].fillna(0).values
        plt.plot(x,y); plt.fill_between(x, y-sd, y+sd, alpha=0.2); plt.title(measure)
        st.pyplot(fig, use_container_width=True)
    except Exception as ex:
        st.error(f"Longitudinal failed: {ex}")

# ----- 5) Response & Safety
with tabs[4]:
    st.header("Response & Safety")
    r = load_csv(p_resp); a = load_csv(p_ae)
    try:
        best = r.groupby("patient_id")["best_pct_change"].min().sort_values()
        fig = plt.figure(figsize=(8,4))
        plt.bar(range(len(best)), best.values); plt.axhline(-30, ls="--"); plt.axhline(20, ls="--")
        plt.ylabel("% change"); plt.title("Waterfall")
        st.pyplot(fig, use_container_width=True)
    except Exception as ex:
        st.error(f"Waterfall failed: {ex}")
    st.write("AE counts", a["ae_term"].value_counts() if "ae_term" in a.columns else a.head())

# ----- 6) QC
with tabs[5]:
    st.header("Diagnostics / QC")
    q = load_csv(p_qc); st.dataframe(q, use_container_width=True)
