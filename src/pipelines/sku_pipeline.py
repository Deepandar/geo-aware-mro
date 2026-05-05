# src/pipelines/sku_pipeline.py

import pandas as pd
import mlflow
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

from src.classifiers.abc_classifier import classify_abc
from src.classifiers.ved_classifier import classify_ved
from src.classifiers.fns_classifier import classify_fns
from src.classifiers.location_scorer import LocationScorer
from src.classifiers.ltr_scorer import LTRScorer
from src.classifiers.criticality_index import CriticalityIndexer
from src.classifiers.dominance_check import DominanceChecker
from src.classifiers.newsvendor import NewsvendorEngine
from src.data_ingestion.synthetic_sku_master import save as generate_data


def run_pipeline():

    cfg = load_config()

    logger = get_logger(
        level=cfg["logging"]["level"],
        fmt=cfg["logging"]["format"]
    )

    input_path = Path(cfg["paths"]["input"])
    output_path = Path(cfg["paths"]["output"])
    auto_generate = cfg["execution"]["auto_generate"]

    logger.info(f"Pipeline start | env config loaded")

    if not input_path.exists():
        if auto_generate:
            logger.warning("Input missing → generating synthetic data")
            input_path.parent.mkdir(parents=True, exist_ok=True)
            generate_data(str(input_path))
        else:
            raise FileNotFoundError(f"Missing input: {input_path}")

    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):

        df = pd.read_csv(input_path)

        df["annual_consumption_value"] = df["unit_cost"] * df["demand"]

        logger.info("Dominance check")
        df, dom = DominanceChecker().check_and_remediate(df)

        df = classify_abc(df)
        df = classify_ved(df)
        df = classify_fns(df)

        df = LocationScorer().score(df)
        df = LTRScorer().compute(df)

        df = CriticalityIndexer().compute(df)

        df = NewsvendorEngine().compute(df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        mlflow.log_artifact(str(output_path))

        logger.info(f"Pipeline complete | output={output_path}")

        return df


if __name__ == "__main__":
    run_pipeline()
