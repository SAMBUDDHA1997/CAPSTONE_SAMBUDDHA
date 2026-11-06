"""Rebuild the chosen classifier on the same 840 rows and verify serialization."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments


def main():
    Path("models").mkdir(exist_ok=True)
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    df = pd.read_csv("data/modelling_table.csv")
    X, y = df[FEATURES], df.is_delayed
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    _, Xtest, _, _ = train_test_split(
        Xtmp, ytmp, test_size=0.5, random_state=42, stratify=ytmp
    )
    depth = None if str(config.max_depth) == "unlimited" else int(config.max_depth)
    pipe = Pipeline(
        [
            ("selector", SelectKBest(f_classif, k=int(config.selector_k))),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=int(config.n_estimators),
                    max_depth=depth,
                    min_samples_leaf=int(config.min_samples_leaf),
                    random_state=42,
                ),
            ),
        ]
    ).fit(Xtr, ytr)
    joblib.dump(pipe, "models/delay_model.pkl")
    pd.DataFrame(dict(threshold=[config.threshold])).to_csv(
        "models/delay_threshold.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame(dict(feature=FEATURES)).to_csv("models/feature_order.csv", index=False)
    _, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
    pd.DataFrame(medians.items(), columns=["column", "median"]).to_csv(
        "models/impute_values.csv", index=False, float_format="%.4f"
    )
    loaded = joblib.load("models/delay_model.pkl")
    diff = np.abs(
        pipe.predict_proba(Xtest)[:, 1] - loaded.predict_proba(Xtest)[:, 1]
    ).max()
    pd.DataFrame(
        dict(
            metric=["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
            value=[len(Xtest), diff, config.threshold],
        )
    ).to_csv("outputs/part_d_reload_check.csv", index=False, float_format="%.4f")
    assert diff <= 1e-9
    print("Model packaged; reload probability difference:", diff)


if __name__ == "__main__":
    main()
