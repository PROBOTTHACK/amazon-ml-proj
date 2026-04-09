import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_raw():
    """Synthetic Amazon deforestation data (2004–2019, 9 Brazilian states)."""
    np.random.seed(42)
    states = ["Pará", "Mato Grosso", "Amazonas", "Rondônia",
              "Maranhão", "Tocantins", "Acre", "Roraima", "Amapá"]
    years = list(range(2004, 2020))
    base = {"Pará": 6000, "Mato Grosso": 5500, "Amazonas": 3000,
            "Rondônia": 2500, "Maranhão": 2000, "Tocantins": 1200,
            "Acre": 800, "Roraima": 600, "Amapá": 200}
    rows = []
    for s in states:
        for y in years:
            d = max(50, np.exp(-0.1*(y-2004)) * base[s] + np.random.normal(0, base[s]*0.1))
            rows.append({
                "state": s, "year": y,
                "deforestation_km2": round(d, 1),
                "fire_count": int(d * np.random.uniform(0.3, 0.8)),
                "rainfall_anomaly_mm": round(np.random.normal(-0.2, 0.8), 2),
                "temp_anomaly_c": round(np.random.normal(0.4, 0.3), 2),
                "agri_expansion_km2": round(d * np.random.uniform(0.4, 0.9), 1),
            })
    df = pd.DataFrame(rows)
    # Inject some dirty data for preprocessing demo
    dirty_idx = np.random.choice(df.index, 15, replace=False)
    df.loc[dirty_idx[:5], "deforestation_km2"] = np.nan
    df.loc[dirty_idx[5:10], "fire_count"] = -99       # invalid sentinel
    df.loc[dirty_idx[10:], "rainfall_anomaly_mm"] = 999  # outlier
    return df


@st.cache_data
def get_clean():
    df = load_raw().copy()
    # 1. Drop rows with missing deforestation
    df = df.dropna(subset=["deforestation_km2"])
    # 2. Fix invalid fire_count sentinels
    df.loc[df["fire_count"] < 0, "fire_count"] = np.nan
    df["fire_count"] = df["fire_count"].fillna(df["fire_count"].median())
    # 3. Cap rainfall outliers at 3 std
    cap = df["rainfall_anomaly_mm"].mean() + 3 * df["rainfall_anomaly_mm"].std()
    df["rainfall_anomaly_mm"] = df["rainfall_anomaly_mm"].clip(upper=cap)
    return df


@st.cache_data
def get_features():
    """Aggregate per-state features for clustering."""
    df = get_clean()
    agg = df.groupby("state").agg(
        mean_deforestation   = ("deforestation_km2",   "mean"),
        deforestation_trend  = ("deforestation_km2",   lambda x: round(np.polyfit(range(len(x)), x, 1)[0], 2)),
        mean_fire_count      = ("fire_count",           "mean"),
        mean_rainfall_anomaly= ("rainfall_anomaly_mm", "mean"),
        mean_temp_anomaly    = ("temp_anomaly_c",       "mean"),
        mean_agri_expansion  = ("agri_expansion_km2",  "mean"),
    ).reset_index()

    feature_cols = [c for c in agg.columns if c != "state"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg[feature_cols])
    return agg, X_scaled, feature_cols