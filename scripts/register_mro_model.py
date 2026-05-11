import mlflow.pyfunc
from stable_baselines3 import PPO
import pandas as pd
import mlflow

def register_model(run_id, artifact_path):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)

    class MROAgentWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.model = PPO.load(context.artifacts["sb3_model"])
        def predict(self, context, model_input):
            # model_input: [Inventory, Health, Distance, Trade_Flow]
            actions, _ = self.model.predict(model_input.values)
            return pd.Series(actions)

    with mlflow.start_run(run_id=run_id):
        mlflow.pyfunc.log_model(
            artifact_path="mro_model_api",
            python_model=MROAgentWrapper(),
            artifacts={"sb3_model": local_path}
        )

if __name__ == "__main__":
    register_model("b8254d5f4e1d4b8e8a62df7e331623b2", "ppo_geo_mro_v1.zip")
