"""Shared preprocessing for historical and not-yet-delivered shipments."""

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]
IMPUTE_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    """Return a cleaned copy and the three fitted or supplied imputation medians."""
    df = raw.drop_duplicates().copy()
    df["service_type"] = df["service_type"].str.strip().str.title()
    if not df["service_type"].isin(PROMISED_HOURS).all():
        raise ValueError("Unknown service type")
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    df["declared_value_inr"] = pd.to_numeric(
        df.pop("declared_value")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="raise",
    )
    df["dispatch_time"] = pd.to_datetime(
        df["dispatch_time"], format="%Y-%m-%d %H:%M", errors="raise"
    )
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    if medians is None:
        medians = {col: float(df[col].median()) for col in IMPUTE_COLUMNS}
    else:
        if set(medians) != set(IMPUTE_COLUMNS):
            raise ValueError("Exactly three imputation medians are required")
        medians = {col: float(medians[col]) for col in IMPUTE_COLUMNS}
    if not all(np.isfinite(list(medians.values()))):
        raise ValueError("Imputation medians must be finite")
    df[IMPUTE_COLUMNS] = df[IMPUTE_COLUMNS].fillna(medians)
    return df, medians


def build_features(clean, hubs):
    """Attach hub metadata and dispatch-time features; never read the outcome."""
    df = clean.merge(hubs, how="left", on="hub_id", sort=False, validate="many_to_one")
    if df["hub_tier"].isna().any():
        raise ValueError("Shipment has an unknown hub_id")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = df["payment_mode"].eq("COD").astype(int)
    df["is_weekend"] = df["dispatch_time"].dt.dayofweek.ge(5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    if df[FEATURES].isna().any().any():
        raise ValueError("Missing model feature after preprocessing")
    return df
