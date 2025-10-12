
import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Natera Oncology Demo", layout="wide")

# ---- Theme (Tempus/Flatiron vibe with your palette)
PALETTE = {"blue_main": "#60A4BF", "green": "#7EB86E", "blue_100": "#CEDFFB", "blue_200": "#A9CCF5", "blue_500": "#329ADB", "teal_800": "#1B5F7F", "gray_500": "#8D97A1"}
ONCOPRINT_COLORS = {"AMP": "#E31A1C", "HOMDEL": "#1F78B4", "MISSENSE": "#008000", "TRUNC": "#000000", "INFRAME": "#708090", "SPLICE": "#FF8C00", "FUSION": "#6A3D9A"}

st.markdown(f"""
<style>
:root {
  --brand-primary: {PALETTE['blue_500']};
  --brand-secondary: {PALETTE['teal_800']};
  --brand-accent: {PALETTE['green']};
  --bg: #ffffff;
  --subtle: {PALETTE['blue_100']};
  --muted: {PALETTE['blue_200']};
  --text: #0f172a;
  --text-muted: {PALETTE['gray_500']};
}
.stApp {
  background: var(--bg);
}
.block-container { padding-top: 0.6rem; }
/* Cards */
.card {
  border: 1px solid #e5e7eb; border-radius: 14px; background: #fff;
  box-shadow: 0 2px 14px rgba(2,6,23,.06); padding: 14px 16px;
}
.pill {
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
  background: var(--subtle); color: var(--brand-secondary); margin-right:6px;
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] button[role="tab"] {
  border: 1px solid #e5e7eb; background:#fff; margin-right:8px; padding:8px 14px; border-radius:12px;
}
.stTabs [aria-selected="true"] { background: var(--brand-primary); color:#fff; border-color: var(--brand-primary); }
h1,h2,h3 { color: var(--text); }
small, .muted { color: var(--text-muted); }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar: data paths
base = Path(__file__).parent / "sample_data"
p_tte = st.sidebar.text_input("Time-to-event CSV", str(base / "timetoevent.csv"))
p_mol = st.sidebar.text_input("Molecular CSV", str(base / "genomics.csv"))
p_tx  = st.sidebar.text_input("Treatments CSV", str(base / "treatments.csv"))
p_long= st.sidebar.text_input("Longitudinal CSV", str(base / "longitudinal.csv"))
p_resp= st.sidebar.text_input("Response visits CSV", str(base / "response_visits.csv"))
p_ae  = st.sidebar.text_input("Adverse events CSV", str(base / "adverse_events.csv"))
p_qc  = st.sidebar.text_input("QC CSV", str(base / "qc.csv"))

@st.cache_data
def load_csv(path): return pd.read_csv(path)

tabs = st.tabs(["Time-to-event","OncoPrint","Treatments","Longitudinal","Response & Safety","QC"])

# ---- 1) Time-to-event (KM basic, cohort split by strata)
with tabs[0]:
    st.header("Kaplan–Meier")
    df = load_csv(p_tte)
    df["start_date"] = pd.to_datetime(df["start_date"]); df["event_date"] = pd.to_datetime(df["event_date"])
    df["time_days"] = (df["event_date"] - df["start_date"]).dt.days.clip(lower=0)
    st.write("First rows", df.head())

    groups = ["All"] + sorted(df["strata"].dropna().unique().tolist())
    pick = st.selectbox("Stratum", groups, index=0)
    sub = df if pick=="All" else df[df["strata"]==pick]
    # quick km
    t = sub.sort_values("time_days")["time_days"].values
    e = sub.sort_values("time_days")["event_observed"].astype(int).values
    uniq = np.unique(t)
    at_risk = len(t); surv_t=[0]; surv_s=[1.0]
    for tt in uniq:
        d = int(e[t==tt].sum())
        if at_risk>0:
            surv_t.append(tt); surv_s.append(surv_s[-1]*(1-d/at_risk))
        at_risk -= int((t==tt).sum())
    fig = plt.figure(); plt.step(surv_t, surv_s, where="post")
    plt.ylim(0,1.05); plt.xlabel("Days"); plt.ylabel("Survival probability")
    st.pyplot(fig, use_container_width=True)

# ---- 2) Molecular OncoPrint (cBioPortal-like colors & bars)
with tabs[1]:
    st.header("OncoPrint")
    df = load_csv(p_mol)
    st.write("First rows", df.head())

    # map events per sample/gene -> a set of tokens
    def event_token(row):
        if row.get("variant_type")=="CNV":
            if str(row.get("cnv_call","")).lower().startswith("amp"): return "AMP"
            if str(row.get("cnv_call","")).lower().startswith("deep"): return "HOMDEL"
            return None
        if row.get("variant_type")=="Fusion": return "FUSION"
        vc = str(row.get("variant_classification","")).lower()
        if "missense" in vc: return "MISSENSE"
        if "nonsense" in vc or "frameshift" in vc: return "TRUNC"
        if "in_frame" in vc or "in frame" in vc: return "INFRAME"
        if "splice" in vc: return "SPLICE"
        return None

    df["event_token"] = df.apply(event_token, axis=1)
    df = df[~df["event_token"].isna()].copy()
    genes = sorted(df["gene"].unique().tolist())[:30]
    samples = sorted(df["sample_id"].unique().tolist())
    # build mapping sample->gene->list of events
    from collections import defaultdict
    grid = {s: {g: [] for g in genes} for s in samples}
    for _,r in df.iterrows():
        if r["gene"] in genes:
            grid[r["sample_id"]][r["gene"]].append(r["event_token"])

    # draw like cBioPortal: stacked glyphs in a cell (max 3)
    fig, ax = plt.subplots(figsize=(max(6,len(genes)*0.35), max(5,len(samples)*0.12)))
    ax.set_xlim(0, len(genes)); ax.set_ylim(0, len(samples))
    ax.set_xticks(np.arange(len(genes))+0.5); ax.set_xticklabels(genes, rotation=90)
    ax.set_yticks(np.arange(len(samples))+0.5); ax.set_yticklabels(samples, fontsize=8)
    ax.invert_yaxis()
    ax.set_facecolor("#f8fafc")

    def draw_cell(x, y, tokens):
        if not tokens: return
        # base light grey background
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="#e5e7eb", edgecolor="#cbd5e1", linewidth=0.5))
        # order tokens for visibility priority
        order = ["AMP","HOMDEL","FUSION","TRUNC","MISSENSE","INFRAME","SPLICE"]
        toks = [t for t in order if t in tokens][:3]
        n = len(toks)
        for i,t in enumerate(toks):
            color = ONCOPRINT_COLORS.get(t, "#111827")
            # top-to-bottom stripes
            h = 1.0/n
            ax.add_patch(plt.Rectangle((x, y + i*h), 1, h, facecolor=color, edgecolor="#e5e7eb", linewidth=0.3))

    for yi,s in enumerate(samples):
        for xi,g in enumerate(genes):
            draw_cell(xi, yi, grid[s][g])

    ax.grid(False); ax.tick_params(length=0)
    st.pyplot(fig, use_container_width=True)

    # top & right barplots (counts) — summary like cBioPortal
    st.subheader("Alteration summaries")
    gene_counts = (df.groupby(["gene","event_token"]).size().unstack(fill_value=0)[list(ONCOPRINT_COLORS.keys())].sum(axis=1).sort_values(ascending=False))
    st.write(pd.DataFrame({"count": gene_counts}))

# ---- 3) Treatments (swimmer mini)
with tabs[2]:
    st.header("Treatment patterns")
    df = load_csv(p_tx)
    df["lot_start_date"] = pd.to_datetime(df["lot_start_date"]); df["lot_end_date"] = pd.to_datetime(df["lot_end_date"])
    pats = df["patient_id"].unique()[:12]
    sdf = df[df["patient_id"].isin(pats)].copy()
    fig = plt.figure(figsize=(8,6))
    y=0; yt=[]; yl=[]
    for pid in pats:
        sub = sdf[sdf["patient_id"]==pid].sort_values("line_number")
        base = sub["lot_start_date"].min()
        for _,r in sub.iterrows():
            x0 = (r["lot_start_date"]-base).days; w = max(1,(r["lot_end_date"]-r["lot_start_date"]).days)
            plt.barh(y, w, left=x0)
            y += 0.4
        yt.append(y-0.2); yl.append(pid); y += 0.6
    plt.yticks(yt, yl); plt.xlabel("Days"); st.pyplot(fig, use_container_width=True)

# ---- 4) Longitudinal
with tabs[3]:
    st.header("Longitudinal / Organ Health")
    df = load_csv(p_long)
    measure = st.selectbox("Measure", sorted(df["measure_name"].unique()))
    sub = df[df["measure_name"]==measure].copy()
    sub["measure_date"] = pd.to_datetime(sub["measure_date"])
    agg = sub.groupby("measure_date")["measure_value"].agg(["mean","std"]).reset_index()
    fig = plt.figure(figsize=(8,4)); x=agg["measure_date"]; y=agg["mean"].values; sd=agg["std"].fillna(0).values
    plt.plot(x,y); plt.fill_between(x, y-sd, y+sd, alpha=0.2); plt.title(measure)
    st.pyplot(fig, use_container_width=True)

# ---- 5) Response & Safety
with tabs[4]:
    st.header("Response & Safety")
    r = load_csv(p_resp); a = load_csv(p_ae)
    best = r.groupby("patient_id")["best_pct_change"].min().sort_values()
    fig = plt.figure(figsize=(8,4)); plt.bar(range(len(best)), best.values); plt.axhline(-30, ls="--"); plt.axhline(20, ls="--")
    plt.ylabel("% change"); plt.title("Waterfall"); st.pyplot(fig, use_container_width=True)
    st.write("AE counts", a["ae_term"].value_counts())

# ---- 6) QC
with tabs[5]:
    st.header("Diagnostics / QC")
    q = load_csv(p_qc); st.dataframe(q, use_container_width=True)
