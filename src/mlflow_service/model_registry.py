import mlflow

from mlflow.tracking import MlflowClient


def get_tracking_uri():

    return mlflow.get_tracking_uri()


def list_experiments():

    client = MlflowClient()

    experiments = client.search_experiments()

    return [
        {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
        }
        for exp in experiments
    ]


def list_registered_models():

    client = MlflowClient()

    models = client.search_registered_models()

    output = []

    for model in models:

        output.append(
            {
                "name": model.name,
                "latest_versions": [
                    v.version
                    for v in model.latest_versions
                ],
            }
        )

    return output
