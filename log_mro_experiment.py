import mlflow
import dvc.api
import sys
import os
from pathlib import Path

# Add the project root (where this script is) to the Python path
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

# 1. Attempt to load the URI from your root config folder
# 2. Fallback to the local sqlite db found in your 'dir' output
try:
    from config.settings import MLFLOW_URI
    print(f"Using MLFLOW_URI from config.settings: {MLFLOW_URI}")
except (ImportError, ModuleNotFoundError):
    MLFLOW_URI = "sqlite:///mlflow.db"
    print(f"Config not found at src.config. Using fallback: {MLFLOW_URI}")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("MRO-Resuscitation-RL")

try:
    # Get the DVC-tracked location hash
    data_url = dvc.api.get_url('data/db/geo_aware_mro.db')
    
    with mlflow.start_run(run_name="4D-Trade-Aware-v1"):
        mlflow.log_param("env_type", "Geo-Aware-Comtrade-Integrated")
        mlflow.log_param("observation_space", "Inventory, Health, Distance, Trade_Flow")
        mlflow.log_param("dvc_hash", data_url)
        
        # Performance metrics from your Stress Test
        mlflow.log_metric("final_ep_rew_mean", 38.7)
        
        # Log the RL Model if it exists
        model_path = "models/ppo_geo_mro_v1.zip"
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)
            print(f"Logged artifact: {model_path}")
        
    print("Successfully logged run to MLflow.")
except Exception as e:
    print(f"Logging failed: {e}")
