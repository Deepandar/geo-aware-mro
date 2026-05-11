from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import mlflow

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
)

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)

from sklearn.metrics import (
    classification_report,
)

from sklearn.preprocessing import (
    LabelEncoder,
)

logger = logging.getLogger(__name__)

RISK_CLASSES = [
    "Low",
    "Medium",
    "High",
    "Critical",
]

RISK_SCORE_MAP = {
    "Low": 0.15,
    "Medium": 0.40,
    "High": 0.70,
    "Critical": 0.95,
}


@dataclass
class QualificationResult:

    n_skus: int

    n_low: int

    n_medium: int

    n_high: int

    n_critical: int

    top_features: list

    tree_depth: int

    best_params: dict

    classification_report: str


class DecisionTreeQualifier:

    HIGH_RISK_COUNTRIES = {
        "RU",
        "UA",
        "BY",
        "IR",
        "KP",
        "SY",
        "MM",
    }

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):

        self.max_depth = max_depth

        self.min_samples_leaf = (
            min_samples_leaf
        )

        self.random_state = (
            random_state
        )

        self.model = None

        self.feature_names = []

        self.le = LabelEncoder()

        logger.info(
            "DecisionTreeQualifier initialised"
        )

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    def _build_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        feats = pd.DataFrame(
            index=df.index
        )

        feats["geo_risk_score"] = (
            df.get(
                "geo_risk_score",
                0.0,
            )
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

        lt = (
            df["lead_time_days"]
            .fillna(30)
            .clip(lower=1)
        )

        lt_std = (
            df.get(
                "std_lead_time",
                lt * 0.2,
            )
            .fillna(
                lt * 0.2
            )
        )

        feats["lead_time_cv"] = (
            (
                lt_std / lt
            )
            .clip(0.0, 2.0)
            / 2.0
        )

        feats["hhi_score"] = (
            df.get(
                "hhi_score",
                0.3,
            )
            .fillna(0.3)
            .clip(0.0, 1.0)
        )

        if (
            "supply_origin_country"
            in df.columns
        ):

            feats["sanctions_flag"] = (
                df[
                    "supply_origin_country"
                ]
                .str.upper()
                .isin(
                    self.HIGH_RISK_COUNTRIES
                )
                .astype(float)
            )

        else:

            feats["sanctions_flag"] = 0.0

        feats["ved_score"] = (
            df.get(
                "ved_score",
                df[
                    "ved_class"
                ].map({
                    "V": 1.0,
                    "E": 0.5,
                    "D": 0.0,
                }),
            )
            .fillna(0.5)
        )

        feats["ltr_score"] = (
            df.get(
                "ltr_score",
                0.0,
            )
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

        feats["ci_score"] = (
            df.get(
                "ci_score",
                0.5,
            )
            .fillna(0.5)
            .clip(0.0, 1.0)
        )

        self.feature_names = (
            list(feats.columns)
        )

        return feats

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    def _generate_labels(
        self,
        df,
        feats,
    ):

        labels = pd.Series(
            ["Medium"] * len(df),
            index=df.index,
            dtype=str,
        )

        geo = feats[
            "geo_risk_score"
        ]

        hhi = feats[
            "hhi_score"
        ]

        ved = feats[
            "ved_score"
        ]

        sxn = feats[
            "sanctions_flag"
        ]

        low_mask = (
            (geo < 0.20)
            & (hhi < 0.30)
            & (sxn == 0)
        )

        labels[
            low_mask
        ] = "Low"

        high_mask = (
            (geo > 0.40)
            | (hhi > 0.60)
            | (sxn == 1)
        )

        labels[
            high_mask
        ] = "High"

        critical_mask = (
            (ved == 1.0)
            & (
                (geo > 0.55)
                | (sxn == 1)
            )
        )

        labels[
            critical_mask
        ] = "Critical"

        return labels

    # -----------------------------------------------------
    # Fit
    # -----------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
    ) -> QualificationResult:

        feats = self._build_features(
            df
        )

        labels = self._generate_labels(
            df,
            feats,
        )

        X = feats.values

        y = labels.values

        y_enc = self.le.fit_transform(
            y
        )

        param_grid = {

            "max_depth": [
                3,
                4,
                5,
                6,
            ],

            "min_samples_leaf": [
                3,
                5,
                8,
            ],

            "criterion": [
                "gini",
                "entropy",
            ],
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state,
        )

        dt = DecisionTreeClassifier(
            random_state=self.random_state
        )

        gs = GridSearchCV(
            dt,
            param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )

        gs.fit(X, y_enc)

        self.model = gs.best_estimator_

        y_pred = self.model.predict(X)

        report = classification_report(
            y_enc,
            y_pred,
            target_names=self.le.classes_,
            zero_division=0,
        )

        importances = list(
            zip(
                self.feature_names,
                self.model.feature_importances_,
            )
        )

        importances.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        dist = labels.value_counts()

        return QualificationResult(

            n_skus=len(df),

            n_low=int(
                dist.get("Low", 0)
            ),

            n_medium=int(
                dist.get("Medium", 0)
            ),

            n_high=int(
                dist.get("High", 0)
            ),

            n_critical=int(
                dist.get("Critical", 0)
            ),

            top_features=importances[:5],

            tree_depth=self.model.get_depth(),

            best_params=gs.best_params_,

            classification_report=report,
        )

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.model is None:

            raise RuntimeError(
                "DecisionTreeQualifier: fit() first."
            )

        out = df.copy()

        feats = self._build_features(
            out
        )

        X = feats.values

        y_enc = self.model.predict(X)

        labels = (
            self.le.inverse_transform(
                y_enc
            )
        )

        out[
            "supplier_risk_class"
        ] = labels

        out[
            "supplier_risk_score"
        ] = [

            RISK_SCORE_MAP.get(
                lbl,
                0.5,
            )

            for lbl in labels
        ]

        out[
            "procurement_flag"
        ] = (
            (
                out[
                    "supplier_risk_class"
                ]
                == "Critical"
            )
            &
            (
                out[
                    "ved_class"
                ]
                == "V"
            )
        )

        return out

    # -----------------------------------------------------
    # Convenience
    # -----------------------------------------------------

    def fit_predict(
        self,
        df,
    ):

        result = self.fit(df)

        out = self.predict(df)

        return out, result

    # -----------------------------------------------------
    # Tree
    # -----------------------------------------------------

    def print_tree(
        self,
    ):

        if self.model is None:

            return (
                "Model not fitted."
            )

        return export_text(
            self.model,
            feature_names=self.feature_names,
        )

    # -----------------------------------------------------
    # MLflow
    # -----------------------------------------------------

    def log_to_mlflow(

        self,

        df,

        result,

        run_name="supplier_qualification_v1.1",

    ):

        with mlflow.start_run(
            run_name=run_name,
            nested=True,
        ):

            mlflow.log_param(
                "tree_depth",
                result.tree_depth,
            )

            mlflow.log_param(
                "best_params",
                str(
                    result.best_params
                ),
            )

            mlflow.log_metric(
                "n_critical",
                result.n_critical,
            )
