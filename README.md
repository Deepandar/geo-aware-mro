# Geo-Aware MRO Decision Intelligence System

**v1.1** · Production-ready decision intelligence pipeline for MRO (Maintenance, Repair & Overhaul) inventory optimization.

A comprehensive supply chain analytics project that combines **multi-criteria SKU classification** (27-class taxonomy), **intermittent demand forecasting**, **Bayesian geo-risk scoring**, **Newsvendor optimization**, and **game-theoretic supplier strategy** into a unified, reproducible decision engine.

---

## 🎯 Objectives

- Build a production-grade, reproducible analytics pipeline for complex MRO spare parts inventory
- Classify SKUs using a 27-class taxonomy (ABC × VED × FNS × Location Criticality)
- Generate demand forecasts tailored to intermittent/lumpy patterns common in aerospace/defense MRO
- Quantify geographic supplier risk using Bayesian updating on real trade, sanctions, and conflict data
- Optimize inventory policies with the Newsvendor model adjusted for geo-risk
- Incorporate strategic supplier intelligence via Decision Trees and Nash Equilibrium modeling
- Deliver production-ready code, dashboards, and documentation as a strong career/portfolio signal

---

## ✨ Key Features (v1.1)

- **27-Class SKU Taxonomy**: ABC (value) × VED (criticality) × FNS (demand pattern) × Location scoring
- **Demand Characterization**: ADI/CV² quadrant classification
- **Forecasting Engine**: Croston’s Method, SBA (Syntetos-Boylan Approximation), Holt-Winters, auto-ARIMA + ensemble routing
- **Bayesian Geo-Risk Layer**: Country-level disruption probability updated with HHI, sanctions, and conflict signals
- **Newsvendor Optimization**: Risk-adjusted reorder quantities and ROP tables
- **Supplier Intelligence**: Decision Tree qualification + Nash Equilibrium strategic risk scoring
- **Interactive Analytics**: Plotly Dash dashboard (classification, forecasts, risk maps)
- **MLOps Foundation**: Full DVC pipelines, MLflow experiment tracking, Dockerized services, pytest suite

---

## 🛠 Tech Stack

- **Core**: Python 3.11, Pandas, NumPy, DuckDB
- **Forecasting**: `sktime`, `pmdarima`, Statsmodels
- **MLOps**: DVC, MLflow, Docker + Docker Compose
- **Visualization**: Plotly, Dash
- **Testing & CI**: pytest, GitHub Actions
- **Documentation**: MkDocs + GitHub Pages

---

## 📁 Repository Structure

```text
geo-aware-mro/
├── data/                  # Raw, processed, and external data (DVC tracked)
├── notebooks/             # Analysis and weekly deliverables
├── src/
│   ├── classifiers/       # ABC, VED, FNS, Location, Supplier DT
│   ├── forecasting/       # Croston, SBA, HW, ARIMA, router
│   ├── risk/              # Bayesian Geo-Risk + HHI
│   ├── optimization/      # Newsvendor, ROP
│   ├── game_theory/       # Nash Equilibrium
│   └── utils/             # Pipeline helpers
├── tests/                 # pytest suite
├── dashboards/            # Plotly Dash app
├── mlruns/                # MLflow tracking
├── dvc.yaml               # Pipeline definitions
├── Dockerfile
├── mkdocs.yml
├── .github/workflows/     # CI/CD
└── docs/                  # Deployed documentation
