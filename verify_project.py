"""Check the brief's acceptance criteria and repeatable data/scoring exports."""

import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, NOISE_COLUMNS, clean_shipments, build_features


def read(name):
    return pd.read_csv("outputs/" + name)


def main():
    checks = read("part_a_checks.csv").set_index("metric").value
    exact = dict(
        raw_rows=1224,
        duplicate_rows_removed=24,
        rows_after_dedup=1200,
        missing_distance_km_before=36,
        missing_parcel_weight_before=24,
        sentinel_weight_rows=5,
        missing_declared_value_before=12,
        missing_payment_mode_before=18,
        service_type_spellings_before=12,
        service_type_spellings_after=3,
        rows_after_hub_merge=1200,
        nulls_in_features_and_target=0,
        delayed_count=278,
        hours_outlier_count=1,
    )
    for key, value in exact.items():
        assert checks[key] == value, (key, checks[key])
    approximate = dict(
        median_distance_km=314.95,
        median_parcel_weight_kg=1.83,
        median_declared_value_inr=1121.5,
        delay_rate=0.2317,
        hours_q1=43.26,
        hours_q3=72.56,
        hours_lower_bound=-0.71,
        hours_upper_bound=116.52,
        declared_value_mean_inr=1533.05,
        declared_value_median_inr=1121.5,
        log_declared_value_mean=7.028,
        log_declared_value_median=7.022,
    )
    for key, value in approximate.items():
        assert abs(checks[key] - value) < 0.01, (key, checks[key])
    groups = read("part_a_delay_by_group.csv")
    assert groups.shipments.tolist() == [
        315,
        243,
        642,
        400,
        400,
        400,
        274,
        563,
        275,
        88,
    ]
    assert groups.delayed.tolist() == [27, 119, 132, 16, 86, 176, 6, 77, 120, 75]
    b = read("part_b_metrics.csv")
    assert len(b) == 6 and sum(b.split.eq("test")) == 1
    assert np.allclose(b.RMSE, np.sqrt(b.MSE), atol=0.001)
    val = b[b.split.eq("val")].set_index("model")
    for model, minimum in dict(
        linear_regression=0.88,
        gradient_boosting=0.88,
        random_forest=0.86,
        decision_tree=0.82,
    ).items():
        assert val.loc[model, "R2"] >= minimum
    assert -0.1 <= val.loc["mean_baseline", "R2"] <= 0.02
    assert b.loc[b.split.eq("test"), "R2"].iloc[0] >= 0.88
    bc = read("part_b_checks.csv").set_index("metric").value
    assert bc.train_rows == 840 and bc.val_rows == 180 and bc.test_rows == 180
    assert bc.train_scaled_max_abs_mean < 1e-9 and bc.test_scaled_max_abs_mean > 0.01
    trees = read("part_b_tree_depth.csv")
    assert len(trees) == 8 and np.all(np.diff(trees.train_rmse) <= 0.0001)
    assert trees.iloc[-1].val_rmse - trees.iloc[-1].train_rmse >= 5
    assert trees.iloc[0].val_rmse - trees.iloc[0].train_rmse <= 2.5
    assert trees.iloc[0].val_rmse - trees.val_rmse.min() >= 4
    assert abs(val.loc["decision_tree", "RMSE"] - trees.val_rmse.min()) < 0.001
    assert trees.iloc[trees.val_rmse.argmin()].max_depth in [
        "4",
        "6",
        "8",
        "12",
        "unlimited",
    ]
    cv = read("part_b_cv.csv")
    assert len(cv) == 2 and (cv.mean_r2 >= 0.9).all() and (cv.std_r2 <= 0.03).all()
    rates = read("part_b_gb_learning_rate.csv").set_index("learning_rate").val_r2
    assert (
        rates[0.1] > rates[1.0] > rates[0.01]
        and rates[0.1] >= 0.9
        and rates[0.01] <= 0.8
    )
    c = read("part_c_models.csv")
    assert len(c) == 4 and c.iloc[0][["TN", "FP", "FN", "TP"]].tolist() == [
        138,
        0,
        42,
        0,
    ]
    assert (c.iloc[1:].accuracy > c.iloc[0].accuracy).all()
    assert (c.iloc[1:].f1 >= 0.6).all()
    assert np.allclose(c.accuracy, (c.TN + c.TP) / 180, atol=0.0001)
    scores = read("part_c_feature_scores.csv")
    top = set(scores.loc[scores.f_rank <= 7, "feature"])
    assert len(scores) == 17 and top == {
        "distance_km",
        "hub_tier",
        "hub_daily_capacity",
        "n_stops",
        "promised_hours",
        "rain_flag",
        "sale_event_flag",
    }
    noise = scores[scores.feature.isin(NOISE_COLUMNS)]
    assert (noise.f_rank >= 8).all() and (noise.f_score < 10).all()
    assert 0.15 <= noise.rf_importance.sum() <= 0.30
    tune = read("part_c_tuning.csv").set_index("search")
    assert (
        tune.loc["grid_search", "candidates"] == 27
        and tune.loc["randomized_search", "candidates"] == 9
    )
    assert (
        tune.loc["grid_search", "total_fits"] == 135
        and tune.loc["randomized_search", "total_fits"] == 45
    )
    assert (
        tune.loc["grid_search", "best_cv_accuracy"]
        >= tune.loc["randomized_search", "best_cv_accuracy"]
    )
    assert (
        tune.best_cv_accuracy.dropna().between(0.86, 0.93).all()
        and (tune.val_f1 >= 0.65).all()
    )
    assert len(read("part_c_selectk.csv")) == 5
    thresholds = read("part_c_thresholds.csv")
    assert len(thresholds) == 9 and (np.diff(thresholds.recall) <= 0).all()
    eligible = thresholds.loc[thresholds.recall >= 0.8, "threshold"]
    expected = eligible.max() if len(eligible) else 0.1
    config = read("part_c_final_config.csv").iloc[0]
    assert config.threshold == expected
    final = read("part_c_final_test.csv").iloc[0]
    assert (
        final.threshold == expected
        and final.f1 >= 0.6
        and final.recall >= 0.55
        and final.precision >= 0.45
    )
    profile = read("hub_profile.csv")
    assert (
        len(profile) == 24
        and profile.hub_id.is_monotonic_increasing
        and profile.shipments.eq(50).all()
    )
    sep = read("part_d_separation.csv")
    assert sep.cluster_sizes.tolist() == ["15-9", "9-8-7", "8-7-5-4", "8-7-4-3-2"]
    assert np.allclose(sep.across_over_within, [2.894, 5.393, 6.305, 6.887], atol=0.01)
    seg = read("segment_profile.csv")
    assert seg.hubs.tolist() == [7, 8, 9] and np.allclose(
        seg.delay_rate, [0.0257, 0.155, 0.46], atol=0.0001
    )
    assert set(read("tier_vs_segment_mismatch.csv").hub_id) == {
        "HUB07",
        "HUB13",
        "HUB15",
        "HUB22",
    }
    reload = read("part_d_reload_check.csv").set_index("metric").value
    assert (
        reload.test_rows_scored == 180
        and reload.reload_max_abs_probability_diff <= 1e-9
    )
    assert pd.read_csv("models/feature_order.csv").feature.tolist() == FEATURES
    actions = read("new_shipment_actions.csv")
    assert (
        len(actions) == 48
        and actions.shipment_id.is_unique
        and not actions.isna().any().any()
    )
    assert np.array_equal(
        actions.flagged, actions.delay_probability.ge(expected).astype(int)
    )
    correct = np.where(
        actions.flagged.eq(0),
        "No action",
        np.where(
            actions.segment.eq("Escalate"),
            "Escalate to hub manager",
            "Proactive customer call",
        ),
    )
    assert np.array_equal(correct, actions.action)
    _, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
    clean, _ = clean_shipments(pd.read_csv("data/new_shipments.csv"), medians)
    assert (
        not build_features(clean, pd.read_csv("data/hubs.csv"))[FEATURES]
        .isna()
        .any()
        .any()
    )
    data_paths = [
        Path("data") / name
        for name in ["hubs.csv", "raw_shipments.csv", "new_shipments.csv"]
    ]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = [digest(p) for p in data_paths]
    subprocess.run([sys.executable, "generate_data.py"], check=True)
    assert before == [digest(p) for p in data_paths]
    path = Path("outputs/new_shipment_actions.csv")
    before = digest(path)
    subprocess.run([sys.executable, "predict_delay.py"], check=True)
    assert before == digest(path)
    print(
        "PASS: reference values, pass bands, consistency, regeneration and repeat scoring."
    )


if __name__ == "__main__":
    main()
