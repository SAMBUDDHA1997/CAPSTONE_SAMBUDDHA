"""Delivery-hours regression: training fits, validation choices, one test evaluation."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES


def save(rows, name):
    pd.DataFrame(rows).to_csv("outputs/" + name, index=False, float_format="%.4f")


def metrics(name, split, y, prediction):
    mse = mean_squared_error(y, prediction)
    return dict(
        model=name,
        split=split,
        MAE=mean_absolute_error(y, prediction),
        MSE=mse,
        RMSE=np.sqrt(mse),
        R2=r2_score(y, prediction),
    )


def main():
    df = pd.read_csv("data/modelling_table.csv")
    X, y = df[FEATURES], df.delivery_hours
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=42)
    Xval, Xtest, yval, ytest = train_test_split(
        Xtmp, ytmp, test_size=0.5, random_state=42
    )
    scaler = StandardScaler().fit(Xtr)
    save(
        [
            dict(metric=k, value=v)
            for k, v in dict(
                train_rows=len(Xtr),
                val_rows=len(Xval),
                test_rows=len(Xtest),
                train_scaled_max_abs_mean=np.abs(
                    scaler.transform(Xtr).mean(axis=0)
                ).max(),
                test_scaled_max_abs_mean=np.abs(
                    scaler.transform(Xtest).mean(axis=0)
                ).max(),
            ).items()
        ],
        "part_b_checks.csv",
    )
    results = [metrics("mean_baseline", "val", yval, np.repeat(ytr.mean(), len(yval)))]
    trees = []
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    for depth in depths:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(Xtr, ytr)
        trees.append(
            dict(
                max_depth="unlimited" if depth is None else depth,
                train_rmse=np.sqrt(mean_squared_error(ytr, model.predict(Xtr))),
                val_rmse=np.sqrt(mean_squared_error(yval, model.predict(Xval))),
            )
        )
    save(trees, "part_b_tree_depth.csv")
    # Input order resolves exact ties toward smaller depth.
    best_index = min(range(len(trees)), key=lambda i: trees[i]["val_rmse"])
    candidates = {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "decision_tree": DecisionTreeRegressor(
            max_depth=depths[best_index], random_state=42
        ),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        results.append(metrics(name, "val", yval, model.predict(Xval)))
    rates = []
    for rate in [0.01, 0.1, 1.0]:
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=rate, random_state=42
        ).fit(Xtr, ytr)
        pred = model.predict(Xval)
        rates.append(
            dict(
                learning_rate=rate,
                val_mse=mean_squared_error(yval, pred),
                val_r2=r2_score(yval, pred),
            )
        )
    save(rates, "part_b_gb_learning_rate.csv")
    ranked = sorted(results[1:], key=lambda row: row["RMSE"])
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    # Reassemble the original training + validation rows; test rows stay excluded.
    Xcv = pd.concat([Xtr, Xval])
    ycv = pd.concat([ytr, yval])
    cv = []
    for row in ranked[:2]:
        scores = cross_val_score(clone(candidates[row["model"]]), Xcv, ycv, cv=folds)
        cv.append(dict(model=row["model"], mean_r2=scores.mean(), std_r2=scores.std()))
    save(cv, "part_b_cv.csv")
    final_name = ranked[0]["model"]
    results.append(
        metrics(final_name, "test", ytest, candidates[final_name].predict(Xtest))
    )
    save(results, "part_b_metrics.csv")
    print("Part B complete; chosen model:", final_name)


if __name__ == "__main__":
    main()
