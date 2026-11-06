"""Run from the repository root: python part_a_clean_eda.py."""

from pathlib import Path
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features


def main():
    Path("outputs").mkdir(exist_ok=True)
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    before = raw.drop_duplicates()
    clean, medians = clean_shipments(raw)
    table = build_features(clean, hubs)
    table["is_delayed"] = (
        table["delivery_hours"].gt(table["promised_hours"]).astype(int)
    )
    table.to_csv("data/modelling_table.csv", index=False)
    q1, q3 = table["delivery_hours"].quantile([0.25, 0.75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    checks = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(before),
        "rows_after_dedup": len(before),
        "missing_distance_km_before": before.distance_km.isna().sum(),
        "missing_parcel_weight_before": before.parcel_weight_kg.isna().sum(),
        "sentinel_weight_rows": before.parcel_weight_kg.gt(100).sum(),
        "missing_declared_value_before": before.declared_value.isna().sum(),
        "missing_payment_mode_before": before.payment_mode.isna().sum(),
        "service_type_spellings_before": before.service_type.nunique(),
        "service_type_spellings_after": clean.service_type.nunique(),
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "rows_after_hub_merge": len(table),
        "nulls_in_features_and_target": table[
            FEATURES + ["delivery_hours", "is_delayed"]
        ]
        .isna()
        .sum()
        .sum(),
        "delayed_count": table.is_delayed.sum(),
        "delay_rate": table.is_delayed.mean(),
        "hours_q1": q1,
        "hours_q3": q3,
        "hours_lower_bound": lower,
        "hours_upper_bound": upper,
        "hours_outlier_count": (
            (table.delivery_hours < lower) | (table.delivery_hours > upper)
        ).sum(),
        "declared_value_mean_inr": table.declared_value_inr.mean(),
        "declared_value_median_inr": table.declared_value_inr.median(),
        "log_declared_value_mean": table.log_declared_value.mean(),
        "log_declared_value_median": table.log_declared_value.median(),
    }
    pd.DataFrame(checks.items(), columns=["metric", "value"]).to_csv(
        "outputs/part_a_checks.csv", index=False, float_format="%.4f"
    )
    table["distance_band"] = pd.cut(
        table.distance_km,
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
    )
    groups = []
    for dimension in ["service_type", "hub_tier", "distance_band"]:
        frame = (
            table.groupby(dimension, observed=True)
            .agg(
                shipments=("shipment_id", "size"),
                delayed=("is_delayed", "sum"),
                delay_rate=("is_delayed", "mean"),
                mean_hours=("delivery_hours", "mean"),
            )
            .reset_index()
            .rename(columns={dimension: "group"})
        )
        frame.insert(0, "dimension", dimension)
        groups.append(frame)
    pd.concat(groups, ignore_index=True).to_csv(
        "outputs/part_a_delay_by_group.csv", index=False, float_format="%.4f"
    )
    correlations = table.select_dtypes(include="number").corr()
    correlations.rename_axis("column").reset_index().to_csv(
        "outputs/part_a_correlations.csv", index=False, float_format="%.4f"
    )
    # Critical acceptance checks catch reconstruction or preprocessing mistakes.
    assert len(table) == 1200 and table.shipment_id.is_unique
    assert (
        checks["delayed_count"] == 278 and checks["nulls_in_features_and_target"] == 0
    )
    assert np.isclose(medians["distance_km"], 314.95)
    new, _ = clean_shipments(pd.read_csv("data/new_shipments.csv"), medians)
    assert len(build_features(new, hubs)) == 48
    print("Part A complete: 1,200 historical rows; 278 delayed; 48 new rows validated.")


if __name__ == "__main__":
    main()
