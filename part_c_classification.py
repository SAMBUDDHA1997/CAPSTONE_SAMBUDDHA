"""Delay classification, feature selection, tuning and validation-only threshold selection."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.shipment_prep import FEATURES, NOISE_COLUMNS


def save(rows, name):
    pd.DataFrame(rows).to_csv("outputs/" + name, index=False, float_format="%.4f")


def evaluate(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        TN=int(tn),
        FP=int(fp),
        FN=int(fn),
        TP=int(tp),
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
    )


def main():
    df = pd.read_csv("data/modelling_table.csv")
    X, y = df[FEATURES], df.is_delayed
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    Xval, Xtest, yval, ytest = train_test_split(
        Xtmp, ytmp, test_size=0.5, random_state=42, stratify=ytmp
    )
    assert [len(Xtr), len(Xval), len(Xtest)] == [840, 180, 180]
    assert [int(ytr.sum()), int(yval.sum()), int(ytest.sum())] == [195, 42, 41]
    rows = [
        dict(
            model="blind_all_on_time", **evaluate(yval, np.zeros(len(yval), dtype=int))
        )
    ]
    logistic = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression())]
    ).fit(Xtr, ytr)
    rows.append(
        dict(model="logistic_regression", **evaluate(yval, logistic.predict(Xval)))
    )
    depths = [2, 3, 4, 5, 6, 8, 10]
    tree_scores = []
    for depth in depths:
        tree = DecisionTreeClassifier(
            criterion="gini", max_depth=depth, random_state=42
        ).fit(Xtr, ytr)
        tree_scores.append((depth, evaluate(yval, tree.predict(Xval))))
    depth, stats = max(tree_scores, key=lambda pair: pair[1]["f1"])
    rows.append(dict(model=f"decision_tree_depth_{depth}", **stats))
    forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    rows.append(dict(model="random_forest", **evaluate(yval, forest.predict(Xval))))
    save(rows, "part_c_models.csv")
    selector = SelectKBest(f_classif, k="all").fit(Xtr, ytr)
    scores = pd.DataFrame(
        dict(
            feature=FEATURES,
            f_score=selector.scores_,
            rf_importance=forest.feature_importances_,
        )
    )
    scores["f_rank"] = scores.f_score.rank(method="first", ascending=False).astype(int)
    scores["is_noise"] = scores.feature.isin(NOISE_COLUMNS)
    save(
        scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]],
        "part_c_feature_scores.csv",
    )
    krows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline(
            [
                ("selector", SelectKBest(f_classif, k=k)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        ).fit(Xtr, ytr)
        krows.append(dict(k=k, val_f1=f1_score(yval, pipe.predict(Xval))))
    save(krows, "part_c_selectk.csv")
    # The brief explicitly fixes selection before CV in this exercise.
    # A production estimate should instead include selection inside each CV fold.
    selected = SelectKBest(f_classif, k=7).fit(Xtr, ytr)
    Atr, Aval = selected.transform(Xtr), selected.transform(Xval)
    grid = dict(
        n_estimators=[50, 100, 150], max_depth=[4, 6, 8], min_samples_leaf=[1, 5, 10]
    )
    searches = [
        (
            "grid_search",
            GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5),
        ),
        (
            "randomized_search",
            RandomizedSearchCV(
                RandomForestClassifier(random_state=42),
                grid,
                n_iter=9,
                cv=5,
                random_state=42,
            ),
        ),
    ]
    tuning = []
    fitted = {}
    for name, search in searches:
        search.fit(Atr, ytr)
        fitted[name] = search.best_estimator_
        tuning.append(
            dict(
                search=name,
                candidates=len(search.cv_results_["params"]),
                cv_folds=5,
                total_fits=len(search.cv_results_["params"]) * 5,
                best_params=str(search.best_params_),
                best_cv_accuracy=search.best_score_,
                val_f1=f1_score(yval, search.predict(Aval)),
            )
        )
    default = RandomForestClassifier(n_estimators=100, random_state=42).fit(Atr, ytr)
    fitted["default_forest_same_7_features"] = default
    tuning.append(
        dict(
            search="default_forest_same_7_features",
            candidates=1,
            cv_folds=0,
            total_fits=1,
            best_params="defaults",
            best_cv_accuracy=np.nan,
            val_f1=f1_score(yval, default.predict(Aval)),
        )
    )
    save(tuning, "part_c_tuning.csv")
    priority = {
        "default_forest_same_7_features": 0,
        "grid_search": 1,
        "randomized_search": 2,
    }
    winner = min(tuning, key=lambda row: (-row["val_f1"], priority[row["search"]]))[
        "search"
    ]
    params = fitted[winner].get_params()
    final = Pipeline(
        [
            ("selector", SelectKBest(f_classif, k=7)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=42,
                ),
            ),
        ]
    ).fit(Xtr, ytr)
    probabilities = final.predict_proba(Xval)[:, 1]
    thresholds = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        thresholds.append(
            dict(threshold=t, **evaluate(yval, (probabilities >= t).astype(int)))
        )
    save(thresholds, "part_c_thresholds.csv")
    eligible = [row["threshold"] for row in thresholds if row["recall"] >= 0.80]
    threshold = max(eligible) if eligible else 0.1
    config = dict(
        selector_k=7,
        n_estimators=params["n_estimators"],
        max_depth="unlimited" if params["max_depth"] is None else params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        threshold=threshold,
        chosen_from=winner,
    )
    save([config], "part_c_final_config.csv")
    # Exactly one test evaluation, after all decisions have been fixed.
    save(
        [
            dict(
                threshold=threshold,
                **evaluate(
                    ytest, (final.predict_proba(Xtest)[:, 1] >= threshold).astype(int)
                ),
            )
        ],
        "part_c_final_test.csv",
    )
    print("Part C complete; chosen configuration:", winner, "threshold:", threshold)


if __name__ == "__main__":
    main()
