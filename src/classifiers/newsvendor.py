# src/classifiers/newsvendor.py

import pandas as pd
import numpy as np
import yaml
from scipy import stats


def load_config():
    with open("config/tsl_config.yaml", "r") as f:
        return yaml.safe_load(f)["tsl_config"]


class NewsvendorEngine:

    def __init__(self):
        cfg = load_config()
        self.tsl_map = cfg["tsl_map"]
        self.fns_mod = cfg["fns_modulation"]

    def resolve_tsl(self, abc, ved, fns):
        low, high = self.tsl_map.get(f"{abc}_{ved}", [0.8, 0.85])
        mod = self.fns_mod.get(fns, 0.5)
        return low + mod * (high - low)

    def _fit_dist(self, mean, std, fns):
        if fns == "F":
            return stats.poisson(mu=max(mean, 0.01)), "poisson"
        elif fns in ["N", "S"]:
            var = std**2
            if var > mean:
                r = mean**2 / (var - mean)
                p = r / (r + mean)
                return stats.nbinom(r, p), "negbin"
        return stats.norm(loc=mean, scale=max(std, 1e-6)), "normal"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["tsl"] = df.apply(
            lambda r: self.resolve_tsl(r["abc_class"], r["ved_class"], r["fns_class"]),
            axis=1
        )

        df["critical_ratio"] = df["tsl"]

        df["mean_demand"] = df["demand"]
        df["std_demand"] = df["demand"] * 0.3

        q_list, rop_list = [], []

        for _, r in df.iterrows():
            dist, _ = self._fit_dist(r["mean_demand"], r["std_demand"], r["fns_class"])
            cr = r["critical_ratio"]

            try:
                q = float(dist.ppf(cr))
            except:
                q = r["mean_demand"]

            z = stats.norm.ppf(cr)

            rop = (
                r["lead_time_days"] * r["mean_demand"] +
                z * np.sqrt(r["lead_time_days"]) * r["std_demand"]
            )

            q_list.append(max(q, 0))
            rop_list.append(max(rop, 0))

        df["q_star"] = q_list
        df["rop"] = rop_list

        return df
