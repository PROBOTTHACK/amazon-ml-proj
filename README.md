# 🌿 Amazon Deforestation Risk Clustering
**SDG 15 · Life on Land | DA-304T Mini ML Project**

---

## What This Project Does

Conservation teams can't treat every Amazon state the same — resources are limited.
This project uses **K-Means Clustering** to automatically group Brazilian Amazon states
into risk profiles based on 6 environmental factors, helping identify where intervention
is needed most.

**Output:** Each state is assigned one of three labels:
- 🔴 High Risk — active deforestation, fires, agricultural pressure
- 🟡 Moderate — mid-level loss, climate stress building
- 🟢 Stable — low deforestation, existing policies working

---

## File Structure

```
forest_v2/
├── app.py                  ← Entry point, page navigation
├── requirements.txt
└── pages/
    ├── data.py             ← Shared data loading & feature engineering
    ├── intro.py            ← Why this project / problem statement
    ├── preprocess.py       ← Cleaning steps + feature matrix
    ├── eda.py              ← 7 exploratory charts
    └── model.py            ← K-Means clustering + results
```

---

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

No dataset download needed — realistic synthetic data is generated automatically.

To use real data: download `def_area_2004_2019.csv` from
[Kaggle](https://www.kaggle.com/datasets/mbogernetto/brazilian-amazon-rainforest-degradation)
and update the `load_raw()` function in `pages/data.py`.

---

## ML Pipeline

```
Raw Data (9 states × 16 years)
    ↓
Preprocessing     →  drop nulls, fix sentinels, cap outliers
    ↓
Feature Engineering →  aggregate to 1 row per state (6 features)
    ↓
StandardScaler    →  normalize so all features contribute equally
    ↓
K-Means (K=3)     →  group states by risk profile
    ↓
Validation        →  Silhouette Score + Elbow Plot + PCA plot
    ↓
Risk Labels       →  High Risk / Moderate / Stable
```

---

## Features Used

| Feature | Description |
|---|---|
| `mean_deforestation` | Average annual forest loss (km²) |
| `deforestation_trend` | Slope over time — positive means worsening |
| `mean_fire_count` | Average annual fire events |
| `mean_rainfall_anomaly` | Deviation from normal rainfall (negative = drier) |
| `mean_temp_anomaly` | Temperature deviation (positive = warmer) |
| `mean_agri_expansion` | Agricultural land growth (km²) |

---

## Why K-Means?

- Only 9 data points (states) — K-Means is the right scale
- Gives hard, interpretable group assignments
- DBSCAN would mark most points as noise at this size
- Hierarchical clustering adds complexity without benefit here

K=3 chosen via **Elbow Method** (inertia flattens after K=3) and confirmed by
**Silhouette Score** peaking at K=3.

---

## Validation

| Method | What it checks |
|---|---|
| Silhouette Score | Are clusters genuinely separated? (>0.5 = good) |
| Elbow Plot | Is K=3 the right number of clusters? |
| PCA Plot | Do clusters look visually separated in 2D? |
| Domain check | Do known hotspots (Pará, Mato Grosso) land in High Risk? |

---

## Tech Stack

| Library | Used for |
|---|---|
| `streamlit` | Multi-page dashboard |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | StandardScaler, KMeans, PCA, silhouette_score |
| `plotly` | All charts |

---

## SDG 15 Connection

> *"Protect, restore and promote sustainable use of terrestrial ecosystems,
> sustainably manage forests, combat desertification, and halt and reverse land degradation."*

This clustering directly supports SDG 15 by turning raw environmental data into
actionable conservation priorities — identifying which states need immediate intervention,
active monitoring, or policy maintenance.
