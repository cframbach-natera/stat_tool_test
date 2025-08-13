# QS Stats Lab (POC) – v3 (regenerated)
This build adds **KPI cards** for Kaplan–Meier and **Preset save/load**.

## New in v3
- **KM KPI cards**: N, Events, and Median survival per cohort (top 6) shown above the curve.
- **Presets**: Save all current settings + filters to a JSON. Load them later to restore the UI.
- Keeps v2 features: at-risk table, log-rank (2+ groups), optional Cox HR (2 groups), two-gene oncoplot w/ Fisher's test.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Use Streamlit Community Cloud; set main file to `app.py`.

## Presets
1. Configure analysis + filters.
2. In **⭐ Presets (beta)**, click **Generate preset** then **Download preset JSON**.
3. Later, **Load preset JSON** to restore settings.
