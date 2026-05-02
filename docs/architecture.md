# Architecture

## Pipeline

DVC → Data Processing → MLflow Tracking → Forecasting Models → Geo-Risk Layer → Inventory Optimization → Dashboard

## Components

- **DVC** for versioning raw and processed data
- **MLflow** for experiment tracking
- **Forecasting models** for intermittent demand
- **Geo-risk layer** for supplier-region risk scoring
- **Dashboard** for decision support
