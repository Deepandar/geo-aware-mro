# Geo-Aware MRO Decision Intelligence System

## Problem Statement

Maintenance, Repair, and Operations (MRO) inventory management faces three critical challenges:

1. **Intermittent Demand**: Spare parts exhibit sporadic, unpredictable demand patterns that traditional forecasting methods (ARIMA, exponential smoothing) fail to capture accurately.

2. **High Stockout Cost**: Critical components have asymmetric costs — holding excess inventory is expensive, but stockouts can ground operations, causing catastrophic downtime.

3. **Geo-Political Supply Risk**: Global supply chains face disruptions from geopolitical tensions, trade restrictions, and regional instabilities that traditional inventory models ignore.

Traditional forecasting approaches assume continuous, stable demand and fail for sparse time series common in MRO contexts.

---

## Objective

Build an end-to-end decision intelligence system that combines:

- **SKU Classification**: Multi-dimensional categorization (ABC × VED × FSN) to prioritize inventory based on value, criticality, and movement velocity
- **Intermittent Demand Forecasting**: Croston's method and Syntetos-Boylan Approximation (SBA) for sparse demand patterns
- **Geo-Risk Quantification**: Bayesian layer incorporating geopolitical risk scores by supplier region
- **Inventory Optimization**: Newsvendor model with risk-adjusted service levels

The system produces actionable recommendations: optimal reorder points, safety stock levels, and supplier diversification strategies.

---

## System Architecture



**Pipeline Flow**:
1. **Data Ingestion**: Historical demand, supplier data, geo-risk indices (DVC-tracked)
2. **Preprocessing**: SKU classification, time series aggregation, feature extraction
3. **Modeling**: Intermittent demand forecasting + geo-risk adjustment (MLflow experiments)
4. **Optimization**: Newsvendor model for reorder points and safety stock
5. **Deployment**: Dockerized API + Streamlit dashboard

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.11 | Core development |
| **Experiment Tracking** | MLflow | Model versioning, hyperparameter tuning |
| **Data Versioning** | DVC | Track datasets, ensure reproducibility |
| **Containerization** | Docker, docker-compose | Environment consistency |
| **CI/CD** | GitHub Actions | Automated testing, quality gates |
| **Forecasting** | statsmodels, scikit-learn | Croston's, SBA, classification |
| **Optimization** | SciPy, NumPy | Newsvendor, inventory optimization |
| **Visualization** | Matplotlib, Seaborn, Streamlit | EDA, dashboard |
| **Testing** | pytest | Unit and integration tests |
| **Documentation** | MkDocs Material | Project docs |

---

## Repository Structure


---

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/Deepandar/geo-aware-mro.git
cd geo-aware-mro
conda create -n geo-mro python=3.11 -y
conda activate geo-mro
pip install -r requirements.txt
```

### 2. Run with Docker

```bash
docker-compose up --build
```

### 3. Run Tests

```bash
pytest
```

### 4. View Documentation

```bash
mkdocs serve
# Open http://127.0.0.1:8000
```

---

## Project Status

**Current Phase**: Week 1 — Infrastructure Setup

- [x] Repository initialization
- [x] DVC setup (data versioning)
- [x] MLflow tracking configuration
- [x] Docker containerization
- [x] CI/CD pipeline (GitHub Actions)
- [x] Documentation scaffold (MkDocs)
- [ ] Data ingestion pipeline (Week 2)
- [ ] SKU classification model (Week 2)
- [ ] Intermittent demand forecasting (Week 3)
- [ ] Geo-risk Bayesian layer (Week 4)
- [ ] Inventory optimization (Week 5)
- [ ] Dashboard deployment (Week 6+)

---

## Contact

**Deepandar Rathore**  
Data Scientist | ML Engineer  
New Delhi, India

GitHub: [@Deepandar](https://github.com/Deepandar)
