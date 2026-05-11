from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    ENVIRONMENT: str = "dev"

    API_KEY: str = "geo-aware-mro-dev"

    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow/mlflow.db"

    class Config:
        env_file = ".env"


settings = Settings()
