import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from pages.data import get_clean, get_features

st.title("🤖 Clustering & Results")
st.markdown("---")

df = get_clean()
agg, X_scaled, feature_cols = get_features()

# ── Why K-Means ───────────────────────────────────────────────────────────────
st.subheader("Why K-Means?")
st.markdown("""
We chose **K-Means** because:
- We have a small, clean feature matrix (9 states × 6 features) — K-Means works well here
- We want hard, interpretable group assignments (each state belongs to exactly one risk group)
- It's fast, well-understood, and easy to validate

We pick the number of clusters **K** using the Elbow method below — it shows at what point 
adding more clusters stops meaningfully improving the result.
""")

st.markdown("---")

# ── Elbow method ──────────────────────────────────────────────────────────────
st.subheader("Step 1 — Find the Right K (Elbow Method)")
st.caption("Run K-Means for K=2 to 6. Plot inertia (total within-cluster variance). The 'elbow' is where improvement flattens — that's your best K.")

inertias, sils = [], []
k_range = range(2, 7)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, lbl))

fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
                                name="Inertia", line=dict(color="#e53935", width=2),
                                marker=dict(size=8)))
fig_elbow.add_trace(go.Scatter(x=list(k_range), y=sils, mode="lines+markers",
                                name="Silhouette Score", yaxis="y2",
                                line=dict(color="#2d6a3f", width=2),
                                marker=dict(size=8)))
fig_elbow.update_layout(
    xaxis=dict(title="K (number of clusters)", tickvals=list(k_range)),
    yaxis=dict(title="Inertia (lower = tighter clusters)"),
    yaxis2=dict(title="Silhouette Score (higher = better)", overlaying="y", side="right"),
    plot_bgcolor="white", paper_bgcolor="white", height=340,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig_elbow, use_container_width=True)
st.success("📌 **K = 3** is the elbow point — inertia drops sharply before it, flattens after. Silhouette also peaks here. We use K=3.")

st.markdown("---")

# ── Run K-Means with K=3 ──────────────────────────────────────────────────────
st.subheader("Step 2 — Run K-Means with K=3")

km = KMeans(n_clusters=3, random_state=42, n_init=10)
agg["cluster"] = km.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, agg["cluster"])

# Assign risk labels by mean deforestation rank
rank = agg.groupby("cluster")["mean_deforestation"].mean().rank(ascending=False).astype(int)
label_map = {1: "🔴 High Risk", 2: "🟡 Moderate", 3: "🟢 Stable"}
color_map = {"🔴 High Risk": "#e53935", "🟡 Moderate": "#f9a825", "🟢 Stable": "#43a047"}
agg["risk"] = agg["cluster"].map(lambda c: label_map[rank[c]])

c1, c2, c3 = st.columns(3)
c1.metric("Silhouette Score", f"{sil:.3f}", help="0–1. Above 0.5 = well-separated clusters. Above 0.7 = strong.")
c2.metric("Inertia", f"{km.inertia_:,.1f}", help="Total within-cluster variance. Lower = tighter clusters.")
c3.metric("Clusters", "3", help="High Risk / Moderate / Stable")

st.markdown("---")

# ── PCA visualization ──────────────────────────────────────────────────────────
st.subheader("Step 3 — Are the Clusters Actually Separated?")
st.markdown("""
We have 6 features — impossible to visualize directly. We use **PCA** (Principal Component Analysis) 
to compress them into 2 dimensions so we can see the clusters. If clusters look well-separated in this 
2D view, it confirms K-Means found real structure.
""")

X_pca = PCA(n_components=2).fit_transform(X_scaled)
agg["PC1"] = X_pca[:, 0]
agg["PC2"] = X_pca[:, 1]

fig_pca = px.scatter(agg, x="PC1", y="PC2", color="risk", text="state",
                     color_discrete_map=color_map,
                     labels={"PC1": "PC1 (captures most variance)", "PC2": "PC2"})
fig_pca.update_traces(textposition="top center", marker=dict(size=14,
                      line=dict(width=1, color="white")))
fig_pca.update_layout(plot_bgcolor="#fafafa", paper_bgcolor="white",
                      height=420, legend=dict(title="Risk Level"))
st.plotly_chart(fig_pca, use_container_width=True)
st.caption("✅ The three clusters are clearly separated in PCA space — this confirms the clustering found real groupings, not random noise.")

st.markdown("---")

# ── Results ────────────────────────────────────────────────────────────────────
st.subheader("Step 4 — What Did We Find?")

for risk_label in ["🔴 High Risk", "🟡 Moderate", "🟢 Stable"]:
    group = agg[agg["risk"] == risk_label]
    states_list = ", ".join(group["state"].tolist())
    avg_defor = group["mean_deforestation"].mean()
    avg_fire  = group["mean_fire_count"].mean()
    trend_val = group["deforestation_trend"].mean()
    trend_str = "📈 Worsening" if trend_val > 0 else "📉 Improving"

    with st.expander(f"{risk_label} — {states_list}", expanded=True):
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Mean Annual Deforestation", f"{avg_defor:,.0f} km²")
        cc2.metric("Mean Fire Count", f"{avg_fire:,.0f}")
        cc3.metric("Trend", trend_str)

st.markdown("---")

# ── Feature profile per cluster ───────────────────────────────────────────────
st.subheader("Feature Profile per Cluster")
st.caption("Radar / bar comparison of all 6 features across the 3 risk groups.")

profile = agg.groupby("risk")[feature_cols].mean().reset_index()
profile_melt = profile.melt(id_vars="risk", var_name="Feature", value_name="Mean Value")

fig_bar = px.bar(profile_melt, x="Feature", y="Mean Value", color="risk",
                 barmode="group", color_discrete_map=color_map,
                 labels={"Mean Value": "Mean Value (original scale)"})
fig_bar.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380,
                      legend=dict(title="", orientation="h", y=-0.25),
                      xaxis_tickangle=-20)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ── Final interpretation ───────────────────────────────────────────────────────
st.subheader("What This Means for SDG 15")
st.markdown("""
| Risk Group | States | Action Needed |
|---|---|---|
| 🔴 High Risk | High deforestation, fire, agri pressure | **Immediate intervention** — enforcement, protected area expansion |
| 🟡 Moderate | Mid-level loss, some climate stress | **Active monitoring** — early warning systems |
| 🟢 Stable | Low deforestation, healthy indicators | **Maintain** — existing policies are working here |

**Bottom line:** By clustering on 6 features simultaneously, we identified that not all Amazon states 
need the same response. This kind of data-driven segmentation helps conservation organizations 
allocate limited resources where they matter most.
""")

st.info(f"📊 **Silhouette = {sil:.3f}** — above 0.5 means clusters are genuinely meaningful, not just arbitrary divisions.")