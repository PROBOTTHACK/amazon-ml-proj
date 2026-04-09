import streamlit as st

st.title("🌍 Why This Project?")
st.markdown("---")

st.markdown("""
### The Problem

Every year, large areas of the Amazon rainforest are destroyed — cleared for agriculture, 
burned intentionally, or degraded by drought. This directly threatens **SDG 15: Life on Land**.

But deforestation doesn't look the same everywhere. Some states are losing forest rapidly, 
others are slowing down, and some are almost stable. **Conservation resources are limited**, 
so the key question is:

> *Which states need urgent intervention, which need monitoring, and which are relatively safe?*

---

### What We're Doing

We use **K-Means Clustering** — an unsupervised machine learning algorithm — to automatically 
group Brazilian Amazon states into **risk profiles** based on multiple environmental factors together.

**Why clustering and not just sorting by deforestation?**  
Because risk isn't just one number. A state might have low deforestation *today* but rising 
temperature anomalies, expanding agriculture, and increasing fire counts — that's a future hotspot. 
Clustering captures this multi-dimensional picture at once.

---

### What We're Finding Out

After clustering, each state gets assigned to a group like:
- 🔴 **High Risk** — actively losing forest fast, high fire activity, agricultural pressure  
- 🟡 **Moderate Risk** — mid-level loss, some climate stress  
- 🟢 **Stable** — low deforestation, relatively healthy indicators  

This tells policymakers: *where to act now, where to watch, and where policies are working.*

---

### How Do We Know It's Correct?

Since this is unsupervised learning (no pre-labelled answers), we validate using:

| Metric | What it checks |
|---|---|
| **Silhouette Score** | Are states in the same cluster more similar to each other than to other clusters? (0–1, higher = better) |
| **Inertia / Elbow Plot** | Does adding more clusters actually improve separation? Helps pick the right K. |
| **Visual PCA Plot** | After reducing 6 features to 2D — do the clusters look clearly separated visually? |
| **Domain sense check** | Do the high-risk states match known real-world hotspots like Pará & Mato Grosso? |

---

### Data

- **Source:** Brazilian Amazon states, 2004–2019  
- **Features used:** deforestation area, trend over time, fire count, rainfall anomaly, temperature anomaly, agricultural expansion  
- **rows:** ~144 (9 states × 16 years)

👉 Use the sidebar to move through the pipeline step by step.
""")