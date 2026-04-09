import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pages.data import get_clean

st.title("📊 Exploratory Data Analysis")
st.markdown("Understanding the data before we model it.")
st.markdown("---")

df = get_clean()

# ── Summary stats ─────────────────────────────────────────────────────────────
st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("States", df["state"].nunique())
c2.metric("Years", df["year"].nunique())
c3.metric("Total Forest Lost", f"{df['deforestation_km2'].sum():,.0f} km²")
c4.metric("Peak Year", str(df.groupby("year")["deforestation_km2"].sum().idxmax()))

st.markdown("---")

# ── Deforestation over time ────────────────────────────────────────────────────
st.subheader("1. How has deforestation changed over time?")
st.caption("Key insight: Deforestation peaked around 2004–2006 and declined after stricter enforcement. But some states are trending up again recently.")

tab1, tab2 = st.tabs(["Total (all states)", "By State"])
with tab1:
    yearly = df.groupby("year")["deforestation_km2"].sum().reset_index()
    fig = px.area(yearly, x="year", y="deforestation_km2",
                  labels={"deforestation_km2": "Total Deforestation (km²)", "year": "Year"},
                  color_discrete_sequence=["#2d6a3f"])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=320)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = px.line(df.groupby(["year","state"])["deforestation_km2"].sum().reset_index(),
                   x="year", y="deforestation_km2", color="state", markers=True,
                   labels={"deforestation_km2": "Deforestation (km²)", "year": "Year"})
    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=320)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Distribution by state ──────────────────────────────────────────────────────
st.subheader("2. Which states are most affected?")
st.caption("Pará and Mato Grosso dominate. The top 2 states account for a disproportionate share of total loss.")

by_state = df.groupby("state")["deforestation_km2"].sum().sort_values(ascending=True).reset_index()
fig3 = px.bar(by_state, x="deforestation_km2", y="state", orientation="h",
              color="deforestation_km2",
              color_continuous_scale=["#c8e6c9", "#e53935"],
              labels={"deforestation_km2": "Total km² Lost"})
fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                   coloraxis_showscale=False, height=340)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Correlation between features ──────────────────────────────────────────────
st.subheader("3. How do features relate to each other?")
st.caption("Fire count is strongly correlated with deforestation — fires are both a cause and a symptom. Agricultural expansion also tracks closely.")

num_cols = ["deforestation_km2", "fire_count", "rainfall_anomaly_mm", "temp_anomaly_c", "agri_expansion_km2"]
corr = df[num_cols].corr()
fig4 = px.imshow(corr, text_auto=".2f",
                 color_continuous_scale=["#1565c0", "white", "#b71c1c"],
                 zmin=-1, zmax=1)
fig4.update_layout(height=380)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Fire vs deforestation ──────────────────────────────────────────────────────
st.subheader("4. Fire activity vs deforestation")
st.caption("Each dot is a state-year. States with more deforestation consistently show more fires.")

fig5 = px.scatter(df, x="deforestation_km2", y="fire_count", color="state",
                  trendline="ols", trendline_scope="overall",
                  opacity=0.65,
                  labels={"deforestation_km2": "Deforestation (km²)", "fire_count": "Fire Count"})
fig5.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=350)
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ── Climate anomalies ──────────────────────────────────────────────────────────
st.subheader("5. Climate anomalies over time")
st.caption("Rainfall has been trending negative (drier) while temperatures trend upward — both conditions increase fire risk and deforestation pressure.")

climate = df.groupby("year").agg(
    rainfall=("rainfall_anomaly_mm", "mean"),
    temp=("temp_anomaly_c", "mean")
).reset_index()

fig6 = go.Figure()
fig6.add_trace(go.Bar(x=climate["year"], y=climate["rainfall"],
                       name="Rainfall Anomaly (mm)", marker_color="#1565c0", opacity=0.7))
fig6.add_trace(go.Scatter(x=climate["year"], y=climate["temp"],
                           name="Temp Anomaly (°C)", mode="lines+markers",
                           line=dict(color="#e53935", width=2), yaxis="y2"))
fig6.update_layout(
    yaxis=dict(title="Rainfall Anomaly (mm)"),
    yaxis2=dict(title="Temp Anomaly (°C)", overlaying="y", side="right"),
    xaxis=dict(title="Year"),
    plot_bgcolor="white", paper_bgcolor="white", height=340,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ── Distribution of deforestation ─────────────────────────────────────────────
st.subheader("6. Distribution of deforestation values")
st.caption("The distribution is right-skewed — most readings are low, but a few extreme values pull the mean up. This justifies StandardScaler before clustering.")

fig7 = px.histogram(df, x="deforestation_km2", nbins=30, color_discrete_sequence=["#2d6a3f"],
                    labels={"deforestation_km2": "Deforestation (km²)"})
fig7.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=300)
st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")

# ── Box plots ──────────────────────────────────────────────────────────────────
st.subheader("7. Spread of deforestation per state")
st.caption("Box plots show median, spread, and outlier years per state. Pará has the widest spread — its worst years are extreme.")

fig8 = px.box(df, x="state", y="deforestation_km2", color="state",
              labels={"deforestation_km2": "Deforestation (km²)", "state": ""},
              color_discrete_sequence=px.colors.qualitative.Safe)
fig8.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                   showlegend=False, height=360)
st.plotly_chart(fig8, use_container_width=True)

st.info("✅ EDA complete. Key takeaways: high variance across states, strong fire-deforestation correlation, worsening climate — all justify a multi-feature clustering approach rather than ranking by a single metric.")