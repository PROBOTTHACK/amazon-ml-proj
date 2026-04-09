import streamlit as st
import pandas as pd
import numpy as np
from pages.data import load_raw, get_clean, get_features

st.title("🔧 Data & Preprocessing")
st.markdown("---")

raw = load_raw()
clean = get_clean()
agg, X_scaled, feature_cols = get_features()

# ── Raw data overview ─────────────────────────────────────────────────────────
st.subheader("Raw Data")
st.markdown(f"**Shape:** {raw.shape[0]} rows × {raw.shape[1]} columns")
st.dataframe(raw.head(20), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", raw.shape[0])
col2.metric("Missing Values", int(raw.isnull().sum().sum()))
col3.metric("Columns", raw.shape[1])

# ── Issues found ──────────────────────────────────────────────────────────────
st.subheader("Issues Found in Raw Data")

issues = {
    "Missing deforestation_km2": int(raw["deforestation_km2"].isna().sum()),
    "Invalid fire_count (< 0)": int((raw["fire_count"] < 0).sum()),
    "Outlier rainfall (> 3 std)": int((raw["rainfall_anomaly_mm"] > raw["rainfall_anomaly_mm"].mean() + 3*raw["rainfall_anomaly_mm"].std()).sum()),
}

for issue, count in issues.items():
    st.warning(f"⚠️ **{issue}** — {count} rows affected")

# ── Steps taken ───────────────────────────────────────────────────────────────
st.subheader("Cleaning Steps")

with st.expander("Step 1 — Drop rows with missing deforestation", expanded=True):
    st.markdown("""
    Deforestation is our core target variable. Rows missing it can't be imputed meaningfully 
    (it's not a random missingness — it likely means no satellite reading was taken).
    
    **Action:** `df.dropna(subset=['deforestation_km2'])`  
    **Rows removed:** 5
    """)

with st.expander("Step 2 — Fix invalid fire_count sentinel values", expanded=True):
    st.markdown("""
    Fire count values of `-99` are clearly invalid data entry errors (sentinel values).  
    We can't drop these rows since all other columns are fine.
    
    **Action:** Replace `-99` with `NaN`, then fill with column median.  
    **Rows fixed:** 5
    """)

with st.expander("Step 3 — Cap rainfall outliers at 3 standard deviations", expanded=True):
    st.markdown("""
    Values of `999` in rainfall anomaly are clearly erroneous. Rather than dropping, 
    we clip extreme values at mean ± 3σ (standard practice for environmental data).
    
    **Action:** `df['rainfall'].clip(upper = mean + 3*std)`  
    **Rows capped:** 5
    """)

# ── Clean data ────────────────────────────────────────────────────────────────
st.subheader("After Cleaning")
col1, col2, col3 = st.columns(3)
col1.metric("Rows Remaining", clean.shape[0], delta=f"{clean.shape[0]-raw.shape[0]} from raw")
col2.metric("Missing Values", int(clean.isnull().sum().sum()))
col3.metric("Outliers Fixed", 10)

st.dataframe(clean.describe().round(2), use_container_width=True)

# ── Feature engineering ───────────────────────────────────────────────────────
st.subheader("Feature Engineering")
st.markdown("""
Raw data has one row per **state × year**. For clustering we need one row per **state** 
with summary features. We aggregate 16 years of data into 6 features:
""")

feat_desc = pd.DataFrame({
    "Feature": feature_cols,
    "How Computed": [
        "Mean of yearly deforestation (km²)",
        "Slope of deforestation over years (positive = getting worse)",
        "Mean annual fire count",
        "Mean rainfall deviation from normal (negative = drier)",
        "Mean temperature deviation (positive = warmer)",
        "Mean agricultural land expansion (km²)",
    ],
    "Why It Matters": [
        "Overall scale of damage",
        "Is the state improving or worsening?",
        "Fires both cause and signal deforestation",
        "Drought increases fire risk",
        "Rising temps stress ecosystems",
        "Agri expansion is the #1 driver of deforestation",
    ]
})
st.dataframe(feat_desc, use_container_width=True, hide_index=True)

st.subheader("Feature Matrix (after aggregation)")
st.markdown(f"Shape: **{agg.shape[0]} states × {len(feature_cols)} features** — this is what goes into clustering.")
st.dataframe(agg.round(2), use_container_width=True, hide_index=True)

st.subheader("Scaling (StandardScaler)")
st.markdown("""
K-Means uses Euclidean distance — so a feature with large values (deforestation in thousands of km²) 
would dominate over a small feature (temp anomaly in tenths of degrees). 

We apply **StandardScaler**: each feature becomes mean=0, std=1. Now all features contribute equally.
""")

scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
scaled_df.insert(0, "state", agg["state"].values)
st.dataframe(scaled_df.round(3), use_container_width=True, hide_index=True)