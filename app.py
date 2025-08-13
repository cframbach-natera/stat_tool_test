
import io, json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# Plots & analysis
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.stats import fisher_exact

st.set_page_config(page_title="QS Stats Lab (POC)", page_icon="📈", layout="wide")

# ---------- Theming tweaks ----------
CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
div[data-testid="stSidebar"] { min-width: 360px; }
h1, h2, h3 { letter-spacing: 0.2px; }
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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Sidebar: analysis selector ----------
analysis = st.sidebar.radio(
    "Analysis",
    ["Kaplan–Meier", "Co-occurrence"],
    help="Select which analysis to run.",
    key="analysis_choice"
)

# ---------- Sample data helpers ----------
def load_sample_km() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    # Cohorts: cancer stage, gene mutation, subtype
    stage = np.random.choice(["I", "II", "III", "IV"], size=n, p=[0.25,0.30,0.30,0.15])
    gene = np.random.choice(["EGFR_mut", "KRAS_mut", "TP53_mut", "WT"], size=n, p=[0.25,0.25,0.30,0.20])
    subtype = np.random.choice(["A", "B", "C"], size=n, p=[0.4,0.4,0.2])

    # Survival: different hazards by cohort to make curves distinct
    base = 0.08
    lam = np.full(n, base, dtype=float)
    lam += np.where(stage=="III", 0.015, 0)
    lam += np.where(stage=="IV", 0.035, 0)
    lam += np.where(gene=="EGFR_mut", -0.015, 0)
    lam += np.where(gene=="KRAS_mut", 0.02, 0)

    durations = np.random.exponential(1/lam).astype(float)
    censor_time = 36.0
    observed = (durations <= censor_time).astype(int)
    durations = np.minimum(durations, censor_time)

    age = np.random.normal(60, 10, size=n).round(1)
    sex = np.random.choice(["F", "M"], size=n)

    df = pd.DataFrame({
        "time": durations.round(2),
        "event": observed,
        "stage": stage,
        "gene": gene,
        "subtype": subtype,
        "age": age,
        "sex": sex
    })
    return df

def load_sample_cooc() -> pd.DataFrame:
    np.random.seed(0)
    n = 350
    # Gene mutation status (0/1), with some realistic patterns (EGFR/KRAS mutual exclusivity)
    EGFR = np.random.binomial(1, 0.22, size=n)
    KRAS = np.where(EGFR==1, 0, np.random.binomial(1, 0.28, size=n))  # discourage co-mutation
    TP53 = np.random.binomial(1, 0.35, size=n)
    PIK3CA = np.random.binomial(1, 0.12, size=n)
    ALK = np.where((EGFR==0) & (KRAS==0), np.random.binomial(1, 0.08, size=n), 0)

    # Additional binary features for matrix/network
    A = np.random.binomial(1, 0.45, size=n)
    B = (A & (np.random.rand(n) < 0.6)).astype(int)
    C = np.random.binomial(1, 0.35, size=n)
    D = ((~A.astype(bool)) & (np.random.rand(n) < 0.3)).astype(int)

    region = np.random.choice(["NA", "EMEA", "APAC", "LATAM"], size=n, p=[0.4,0.25,0.25,0.1])
    score = np.random.normal(50, 15, size=n).round(1)
    cohort = np.random.choice(["2023", "2024", "2025"], size=n)

    df = pd.DataFrame({
        "EGFR": EGFR, "KRAS": KRAS, "TP53": TP53, "PIK3CA": PIK3CA, "ALK": ALK,
        "A": A, "B": B, "C": C, "D": D,
        "region": region, "score": score, "cohort": cohort
    })
    # Give each sample an ID column for oncoplot
    df.insert(0, "sample_id", [f"S{i:03d}" for i in range(1, n+1)])
    return df

# ---------- Upload or sample ----------
def file_section(label: str):
    with st.sidebar.expander("📤 Data", expanded=True):
        use_sample = st.toggle(
            "Use sample dataset",
            value=True if analysis=="Kaplan–Meier" else False,
            help="Try a built-in sample if you don't have a file handy.",
            key="use_sample_toggle"
        )
        file = st.file_uploader(label, type=["csv"], accept_multiple_files=False, key="csv_uploader")
        df = None
        sample_name = None
        if use_sample and file is None:
            if analysis == "Kaplan–Meier":
                df = load_sample_km()
                sample_name = "sample_kaplan_meier.csv"
            else:
                df = load_sample_cooc()
                sample_name = "sample_cooccurrence.csv"
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("Download sample CSV", data=buf.getvalue().encode("utf-8"),
                               file_name=sample_name, mime="text/csv", key="dl_sample")
        elif file is not None:
            df = pd.read_csv(file)
        return df

# ---------- Presets (save & load) ----------
ALLOWED_KEYS_STATIC = [
    # app-level
    "analysis_choice",
    # KM
    "km_time_col","km_event_col","km_cohort_col","km_include_levels",
    "km_show_ci","km_show_legend","km_show_at_risk","km_show_logrank","km_show_cox",
    # Co-occurrence
    "cooc_view","cooc_features","cooc_metric","cooc_threshold","cooc_max_edges",
    # Oncoplot
    "oncoplot_gene_a","oncoplot_gene_b","oncoplot_sort",
]

def preset_sidebar(df: pd.DataFrame):
    with st.sidebar.expander("⭐ Presets (beta)", expanded=False):
        st.caption("Save your current settings and filters to a JSON preset, or load a preset.")
        # Load
        up = st.file_uploader("Load preset JSON", type=["json"], key="preset_upload")
        if up is not None:
            try:
                preset = json.loads(up.getvalue().decode("utf-8"))
                sess = preset.get("session", {})
                for k, v in sess.items():
                    st.session_state[k] = v
                st.success("Preset loaded. The UI will refresh with your saved settings.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load preset: {e}")

        st.markdown("---")
        # Save
        name = st.text_input("Preset name", value=st.session_state.get("preset_name","my_preset"), key="preset_name")
        if st.button("Generate preset", type="primary", key="btn_make_preset"):
            preset = build_preset(df, name=name)
            st.session_state["last_preset_json"] = json.dumps(preset, indent=2)
        if "last_preset_json" in st.session_state:
            st.download_button("Download preset JSON",
                               data=st.session_state["last_preset_json"].encode("utf-8"),
                               file_name=f"qs_stats_preset_{name}.json",
                               mime="application/json",
                               key="dl_preset")

def build_preset(df: pd.DataFrame, name: str = "preset") -> Dict:
    # Collect dynamic filter keys
    dyn_keys = [k for k in st.session_state.keys() if k.startswith("flt_")]
    keep = {k: st.session_state[k] for k in ALLOWED_KEYS_STATIC if k in st.session_state}
    keep.update({k: st.session_state[k] for k in dyn_keys if k in st.session_state})
    preset = {
        "meta": {
            "name": name,
            "created": datetime.utcnow().isoformat() + "Z",
            "columns": df.columns.tolist()
        },
        "session": keep
    }
    return preset

# ---------- Adaptive filter UI ----------
def build_filters(df: pd.DataFrame) -> Dict:
    filters = {}
    with st.sidebar.expander("🔍 Filters (optional)", expanded=True):
        st.caption("Auto-generated from your columns. Use to subset before analysis.")
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                _min, _max = float(np.nanmin(series)), float(np.nanmax(series))
                if np.isfinite(_min) and np.isfinite(_max) and _min != _max:
                    key = f"flt_{col}_range"
                    default_val = st.session_state.get(key, (float(_min), float(_max)))
                    val = st.slider(f"{col} range", min_value=float(_min), max_value=float(_max),
                                    value=default_val, key=key)
                    filters[col] = ("range", val)
            elif pd.api.types.is_bool_dtype(series):
                key = f"flt_{col}_bool_choice"
                default_choice = st.session_state.get(key, "All")
                choice = st.selectbox(f"{col} filter", ["All", "True only", "False only"], key=key, index=["All", "True only", "False only"].index(default_choice) if default_choice in ["All","True only","False only"] else 0)
                if choice != "All":
                    filters[col] = ("bool", True if choice=="True only" else False)
            else:
                nunique = series.nunique(dropna=True)
                if 1 < nunique <= 50:
                    opts = sorted([x for x in series.dropna().unique().tolist()], key=lambda x: str(x))
                    key = f"flt_{col}_in"
                    default_sel = st.session_state.get(key, opts)
                    sel = st.multiselect(f"{col} is any of", opts, default=default_sel, key=key)
                    if len(sel) != len(opts):
                        filters[col] = ("in", set(sel))
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, (ftype, val) in filters.items():
        if ftype == "range":
            lo, hi = val
            mask &= df[col].astype(float).between(lo, hi)
        elif ftype == "bool":
            mask &= df[col] == val
        elif ftype == "in":
            mask &= df[col].isin(val)
    return df[mask].copy()

# ---------- Utilities ----------
def to_png_download(fig, filename="plot.png"):
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(fig, format="png", engine="kaleido", scale=2)
        st.download_button("Download plot as PNG", data=png_bytes, file_name=filename, mime="image/png")
    except Exception as e:
        st.info("Install `kaleido` for PNG download support. Still showing the interactive figure below.")

def df_download_button(df: pd.DataFrame, name="filtered.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", data=csv, file_name=name, mime="text/csv")

def is_binary_series(s: pd.Series) -> bool:
    vals = pd.Series(s.dropna().unique())
    if len(vals) == 0:
        return False
    normalized = (
        vals.replace({True:1, False:0})
            .astype(str)
            .str.lower()
            .replace({"true":"1","false":"0"})
    )
    return set(normalized.unique()).issubset({"0","1"})

def normalize_binary_series(s: pd.Series) -> pd.Series:
    return (
        s.replace({True:1, False:0})
         .astype(str)
         .str.lower()
         .replace({"true":"1","false":"0"})
         .fillna("0")
         .astype(float)
         .astype(int)
    )

# ---------- KPI renderer ----------
def render_kpis(summary_rows: List[Dict]):
    if not summary_rows:
        return
    # Show up to 6 cohorts as KPI cards
    rows = sorted(summary_rows, key=lambda r: (-r.get("N",0), str(r.get("Cohort",""))))
    top = rows[:6]
    cols = st.columns(len(top))
    for i, r in enumerate(top):
        med = r.get("Median survival", np.nan)
        med_txt = "NA" if (med is None or (isinstance(med, float) and not np.isfinite(med))) else f"{med:.2f}"
        n = r.get("N", 0)
        e = r.get("Events", 0)
        rate = f"{(e/n*100):.0f}%" if n else "—"
        html = f"""
        <div class="kpi-card">
          <div class="kpi-title">{r.get("Cohort","All cohorts")}</div>
          <div class="kpi-value">Median: {med_txt}</div>
          <div class="small">N={n} • Events={e} ({rate})</div>
        </div>
        """
        with cols[i]:
            st.markdown(html, unsafe_allow_html=True)
    if len(rows) > 6:
        with st.expander("More cohorts"):
            st.dataframe(pd.DataFrame(rows))

# ---------- Kaplan–Meier ----------
def km_ui(df: pd.DataFrame):
    st.subheader("Kaplan–Meier Survival Analysis")
    with st.container():
        left, right = st.columns([1.75, 1])
        with right:
            st.markdown("### Configure")
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            cat_like = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c])) or df[c].nunique() <= 50]

            # Required
            time_col = st.selectbox("Time-to-event column", num_cols,
                                    index=0 if "time" in df.columns else (0 if num_cols else 0),
                                    key="km_time_col")
            event_col = st.selectbox("Event indicator (1=event, 0=censored)", num_cols,
                                     index=0 if "event" in df.columns else (1 if len(num_cols)>1 else 0),
                                     key="km_event_col")

            # Cohorts (custom)
            choices = ["(none)"] + cat_like
            default_index = 0
            for preferred in ["group", "stage", "gene", "subtype"]:
                if preferred in cat_like:
                    default_index = choices.index(preferred)
                    break
            cohort_col = st.selectbox("Cohort/strata column (optional)",
                                      choices, index=default_index, key="km_cohort_col")
            include_levels = None
            if cohort_col and cohort_col != "(none)":
                levels = df[cohort_col].dropna().unique().tolist()
                default_lvls = st.session_state.get("km_include_levels", sorted(levels, key=lambda x: str(x))[:min(5,len(levels))])
                include_levels = st.multiselect("Cohort levels to include",
                                                options=sorted(levels, key=lambda x: str(x)),
                                                default=default_lvls,
                                                key="km_include_levels")
            show_ci = st.checkbox("Show confidence interval", value=st.session_state.get("km_show_ci", True), key="km_show_ci")
            show_legend = st.checkbox("Show legend", value=st.session_state.get("km_show_legend", True), key="km_show_legend")
            show_at_risk = st.checkbox("Show at-risk table", value=st.session_state.get("km_show_at_risk", True), key="km_show_at_risk")
            show_logrank = st.checkbox("Log-rank test", value=st.session_state.get("km_show_logrank", True), key="km_show_logrank")
            show_cox = st.checkbox("Cox HR (2 groups)", value=st.session_state.get("km_show_cox", True), key="km_show_cox")

            st.divider()
            st.markdown("#### Notes")
            st.markdown(
                "- The event column should be binary (1=event occurred, 0=censored).\n"
                "- Compare custom cohorts by choosing a column (e.g., cancer stage, gene, subtype) and selecting levels."
            )

        with left:
            if time_col and event_col:
                df_work = df.copy()
                if cohort_col and cohort_col != "(none)" and include_levels:
                    df_work = df_work[df_work[cohort_col].isin(include_levels)].copy()

                # KPI cards
                rows = []
                if cohort_col and cohort_col != "(none)":
                    for g, gdf in df_work.groupby(cohort_col):
                        km = KaplanMeierFitter().fit(gdf[time_col], gdf[event_col])
                        rows.append({
                            "Cohort": str(g),
                            "N": int(len(gdf)),
                            "Events": int(gdf[event_col].sum()),
                            "Median survival": float(km.median_survival_time_) if km.median_survival_time_ is not None else np.nan
                        })
                else:
                    km = KaplanMeierFitter().fit(df_work[time_col], df_work[event_col])
                    rows.append({
                        "Cohort": "All",
                        "N": int(len(df_work)),
                        "Events": int(df_work[event_col].sum()),
                        "Median survival": float(km.median_survival_time_) if km.median_survival_time_ is not None else np.nan
                    })
                render_kpis(rows)
                st.markdown('<hr class="subtle" />', unsafe_allow_html=True)

                # Plot
                fig, ax = plt.subplots(figsize=(9.5, 5.8))
                fitters = []
                if cohort_col and cohort_col != "(none)":
                    for g, gdf in df_work.groupby(cohort_col):
                        km = KaplanMeierFitter()
                        try:
                            km.fit(durations=gdf[time_col], event_observed=gdf[event_col], label=str(g))
                            km.plot_survival_function(ax=ax, ci_show=show_ci)
                            fitters.append(km)
                        except Exception as e:
                            st.warning(f"Could not fit KM for group '{g}': {e}")
                else:
                    km = KaplanMeierFitter()
                    km.fit(durations=df_work[time_col], event_observed=df_work[event_col], label="All")
                    km.plot_survival_function(ax=ax, ci_show=show_ci)
                    fitters.append(km)

                ax.set_xlabel("Time")
                ax.set_ylabel("Survival probability")
                ax.grid(True, alpha=0.3)
                if not show_legend:
                    ax.get_legend().remove() if ax.get_legend() else None

                if show_at_risk and len(fitters) >= 1:
                    try:
                        add_at_risk_counts(*fitters, ax=ax)
                        st.markdown('<div class="at-risk-note">At-risk counts shown below the x-axis.</div>',
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.info(f"At-risk table unavailable: {e}")

                st.pyplot(fig, use_container_width=True)

                # Stats
                with st.expander("📊 Statistics"):
                    if cohort_col and cohort_col != "(none)":
                        st.dataframe(pd.DataFrame(rows))
                    else:
                        st.write(f"Median survival: **{rows[0]['Median survival']:.2f}**")

                    if show_logrank and (cohort_col and cohort_col != "(none)"):
                        groups = [g for g in df_work[cohort_col].dropna().unique()]
                        if len(groups) == 2:
                            g1, g2 = groups[:2]
                            T1 = df_work[df_work[cohort_col]==g1][time_col]
                            E1 = df_work[df_work[cohort_col]==g1][event_col]
                            T2 = df_work[df_work[cohort_col]==g2][time_col]
                            E2 = df_work[df_work[cohort_col]==g2][event_col]
                            res = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
                            st.write(f"Log-rank test ( {g1} vs {g2} ): p = **{res.p_value:.4g}**")
                        elif len(groups) > 2:
                            res = multivariate_logrank_test(df_work[time_col], df_work[cohort_col], event_observed=df_work[event_col])
                            st.write(f"Multigroup log-rank test across {len(groups)} cohorts: p = **{res.p_value:.4g}**")

                    if show_cox and (cohort_col and cohort_col != "(none)"):
                        groups = [g for g in df_work[cohort_col].dropna().unique()]
                        if len(groups) == 2:
                            ref = groups[0]
                            tmp = df_work[[time_col, event_col, cohort_col]].copy()
                            tmp["group_bin"] = (tmp[cohort_col] != ref).astype(int)
                            cph = CoxPHFitter()
                            try:
                                cph.fit(tmp[[time_col, event_col, "group_bin"]], duration_col=time_col, event_col=event_col)
                                hr = float(np.exp(cph.params_["group_bin"]))
                                ci = cph.confidence_intervals_.loc["group_bin"].values
                                st.write(f"Cox PH HR ( {groups[1]} vs {groups[0]} ): **{hr:.3g}** "
                                         f"[{np.exp(ci[0]):.3g}, {np.exp(ci[1]):.3g}]")
                            except Exception as e:
                                st.info(f"Cox model failed: {e}")

    with st.expander("📥 Export"):
        df_download_button(df, name="km_filtered.csv")

# ---------- Co-occurrence & Oncoplot ----------
def build_cooc_matrix(df: pd.DataFrame, features: List[str], metric: str = "Count") -> pd.DataFrame:
    X = df[features].copy()
    for c in features:
        if X[c].dtype != "int64" and X[c].dtype != "float64":
            X[c] = normalize_binary_series(X[c])
        X[c] = (X[c] != 0).astype(int)
    M = np.dot(X.T, X)
    cooc = pd.DataFrame(M, index=features, columns=features).astype(int)
    np.fill_diagonal(cooc.values, 0)

    if metric == "Jaccard":
        col_sums = np.diag(M).reshape(-1, 1)
        union = col_sums + col_sums.T - cooc.values
        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.where(union > 0, cooc.values / union, 0.0)
        cooc = pd.DataFrame(j, index=features, columns=features)
    elif metric == "Lift":
        N = len(X)
        p = np.diag(M) / N
        p_ab = cooc.values / N
        denom = p.reshape(-1,1) * p.reshape(1,-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(denom > 0, p_ab / denom, 0.0)
        cooc = pd.DataFrame(lift, index=features, columns=features)

    return cooc

def two_gene_oncoplot_ui(df: pd.DataFrame):
    st.markdown("### Two-gene Oncoplot + Mutual Exclusivity")
    binary_cols = [c for c in df.columns if is_binary_series(df[c])]
    if "sample_id" in df.columns:
        id_col = "sample_id"
    else:
        df = df.copy()
        df["sample_id"] = [f"S{i:03d}" for i in range(1, len(df)+1)]
        id_col = "sample_id"

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        g1 = st.selectbox("Gene A", options=binary_cols, index=(binary_cols.index("EGFR") if "EGFR" in binary_cols else 0), key="oncoplot_gene_a")
    with col2:
        g2 = st.selectbox("Gene B", options=binary_cols, index=(binary_cols.index("KRAS") if "KRAS" in binary_cols else min(1, len(binary_cols)-1)), key="oncoplot_gene_b")
    with col3:
        sort_mode = st.selectbox("Sort samples by", ["A→B pattern", "Mutation count", "Original order"], key="oncoplot_sort")

    if g1 == g2:
        st.info("Select two different genes.")
        return

    A = normalize_binary_series(df[g1])
    B = normalize_binary_series(df[g2])
    tmp = df[[id_col]].copy()
    tmp[g1] = A
    tmp[g2] = B
    tmp["mut_count"] = A + B
    tmp["pattern_code"] = A*2 + B  # 0: WT/WT, 1: WT/B, 2: A/WT, 3: A/B

    if sort_mode == "A→B pattern":
        tmp = tmp.sort_values(["pattern_code", id_col])
    elif sort_mode == "Mutation count":
        tmp = tmp.sort_values(["mut_count", id_col], ascending=[False, True])

    # Oncoplot
    mat = np.vstack([tmp[g1].values, tmp[g2].values])
    fig = px.imshow(
        mat,
        x=tmp[id_col].tolist(),
        y=[g1, g2],
        labels=dict(x="Sample", y="Gene", color="Mutated"),
        aspect="auto",
        color_continuous_scale=[[0, "rgb(240,240,240)"], [1, "rgb(31,119,180)"]],
        zmin=0, zmax=1
    )
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=260, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Mutual exclusivity (Fisher's exact test)
    a = int(((A==1) & (B==1)).sum())  # both
    b = int(((A==1) & (B==0)).sum())  # A only
    c = int(((A==0) & (B==1)).sum())  # B only
    d = int(((A==0) & (B==0)).sum())  # neither
    table = np.array([[a, b], [c, d]])  # [[A&B, A only],[B only, neither]]
    try:
        OR, p_two = fisher_exact(table, alternative="two-sided")
        _, p_less = fisher_exact(table, alternative="less")      # OR < 1 (exclusivity)
        _, p_greater = fisher_exact(table, alternative="greater")# OR > 1 (co-occur)
    except Exception as e:
        OR, p_two, p_less, p_greater = np.nan, np.nan, np.nan, np.nan

    with st.container():
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Contingency table**")
            st.dataframe(pd.DataFrame(
                [[a, b],[c, d]],
                index=[f"{g1}=1 & {g2}=1", f"{g1}=0 & {g2}=1"],
                columns=[f"{g1}=1 & {g2}=0", f"{g1}=0 & {g2}=0"]
            ))
        with c2:
            st.markdown("**Mutual exclusivity stats (Fisher)**")
            if isinstance(OR, float) and np.isfinite(OR):
                verdict = "Mutual exclusivity" if (OR < 1 and p_less < 0.05) else ("Co-occurrence" if (OR > 1 and p_greater < 0.05) else "No significant association")
                st.write(f"- Odds ratio: **{OR:.3g}**")
                st.write(f"- p (two-sided): **{p_two:.3g}**")
                st.write(f"- p (OR<1, exclusivity): **{p_less:.3g}**  •  p (OR>1, co-occur): **{p_greater:.3g}**")
                st.write(f"- Verdict: **{verdict}**")
            else:
                st.write("Stats unavailable for this selection.")

    with st.expander("📥 Export"):
        to_png_download(fig, filename="oncoplot_two_gene.png")
        st.download_button("Download 2x2 table (CSV)",
                           data=pd.DataFrame([[a,b],[c,d]]).to_csv(index=False, header=False).encode("utf-8"),
                           file_name=f"{g1}_{g2}_contingency.csv",
                           mime="text/csv")

def cooc_ui(df: pd.DataFrame):
    st.subheader("Co-occurrence Analysis")
    with st.container():
        left, right = st.columns([1.2, 1])
        with right:
            st.markdown("### Configure")
            binary_cols = [c for c in df.columns if is_binary_series(df[c])]
            if not binary_cols:
                st.warning("No binary features detected (0/1, True/False). You can still select any numeric columns below.")
                binary_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

            view = st.radio("View", ["Heatmap", "Network", "Two-gene oncoplot"], index=0, key="cooc_view")

            if view in ("Heatmap", "Network"):
                default_feats = [c for c in ["EGFR","KRAS","TP53","PIK3CA","ALK"] if c in binary_cols]
                if not default_feats:
                    default_feats = binary_cols[: min(6, len(binary_cols))]
                features = st.multiselect("Feature columns (0/1)", options=binary_cols, default=default_feats, key="cooc_features")
                metric = st.radio("Association metric", ["Count", "Jaccard", "Lift"], index=0, key="cooc_metric")
                max_thr = 1.0 if metric != "Count" else float(df.shape[0])
                default_thr = 0.2 if metric != "Count" else 5.0
                threshold = st.slider("Edge threshold (for Network)", min_value=0.0, max_value=max_thr, value=float(st.session_state.get("cooc_threshold", default_thr)), key="cooc_threshold")
                max_edges = st.number_input("Max edges (Network)", min_value=10, max_value=2000, value=int(st.session_state.get("cooc_max_edges", 200)), step=10, key="cooc_max_edges")
            else:
                features, metric, threshold, max_edges = None, None, None, None

        with left:
            if view == "Heatmap":
                if len(features or []) >= 2:
                    cooc = build_cooc_matrix(df, features, metric=metric)
                    fig = px.imshow(
                        cooc.values,
                        x=features, y=features,
                        labels=dict(x="Feature", y="Feature", color=metric),
                        color_continuous_scale="Blues",
                        aspect="auto",
                    )
                    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("📥 Export"):
                        to_png_download(fig, filename="cooccurrence_heatmap.png")
                        st.download_button("Download co-occurrence matrix (CSV)",
                                           data=cooc.to_csv().encode("utf-8"),
                                           file_name="cooccurrence_matrix.csv",
                                           mime="text/csv")
                else:
                    st.info("Select at least two feature columns to compute co-occurrence.")
            elif view == "Network":
                if len(features or []) >= 2:
                    cooc = build_cooc_matrix(df, features, metric=metric)
                    G = nx.Graph()
                    for i, a in enumerate(features):
                        G.add_node(a)
                        for j, b in enumerate(features):
                            if j <= i:
                                continue
                            w = float(cooc.loc[a, b])
                            if (metric == "Count" and w >= threshold) or (metric != "Count" and w >= threshold):
                                G.add_edge(a, b, weight=w)

                    if G.number_of_edges() > max_edges:
                        edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)[:max_edges]
                        G = nx.Graph()
                        G.add_nodes_from(features)
                        G.add_edges_from(edges_sorted)

                    pos = nx.spring_layout(G, seed=42, k=None)
                    edge_x, edge_y = [], []
                    for u, v, data in G.edges(data=True):
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        edge_x += [x0, x1, None]
                        edge_y += [y0, y1, None]

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        mode="lines",
                        line=dict(width=1),
                        hoverinfo="none"
                    )

                    node_x = [pos[n][0] for n in G.nodes()]
                    node_y = [pos[n][1] for n in G.nodes()]
                    node_text = [f"{n} (deg {G.degree(n)})" for n in G.nodes()]
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode="markers+text",
                        text=[n for n in G.nodes()],
                        textposition="top center",
                        hovertext=node_text,
                        hoverinfo="text",
                        marker=dict(size=[8 + 2*G.degree(n) for n in G.nodes()])
                    )

                    fig = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(
                                        title="Co-occurrence Network",
                                        showlegend=False,
                                        margin=dict(l=0,r=0,t=40,b=0),
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        height=650
                                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("📥 Export"):
                        to_png_download(fig, filename="cooccurrence_network.png")
                        st.download_button("Download co-occurrence matrix (CSV)",
                                           data=cooc.to_csv().encode("utf-8"),
                                           file_name="cooccurrence_matrix.csv",
                                           mime="text/csv")
                else:
                    st.info("Select at least two feature columns to build the network.")
            else:
                two_gene_oncoplot_ui(df)

    with st.expander("📥 Export"):
        df_download_button(df, name="cooccurrence_filtered.csv")

# ------------------------------
# Main flow
# ------------------------------
df = file_section("Upload CSV for the selected analysis")

# Presets sidebar (after data is available so we can capture filter widgets)
if df is not None and not df.empty:
    preset_sidebar(df)

if df is None or df.empty:
    st.info("Upload a CSV or enable the sample dataset in the sidebar to continue.")
    st.stop()

# Filters
filters = build_filters(df)
df_filtered = apply_filters(df, filters)

# Preview
with st.expander("👀 Data preview", expanded=False):
    st.dataframe(df_filtered.head(50), use_container_width=True)

# Route to analysis
if analysis == "Kaplan–Meier":
    km_ui(df_filtered)
else:
    cooc_ui(df_filtered)

st.sidebar.caption("Built with ❤️ in Streamlit. This is a POC; Redshift/QuickSight integration forthcoming.")
