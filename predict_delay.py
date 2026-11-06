"""Score new shipments using the saved classifier, preprocessing medians and segments."""

import joblib
import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features


def main():
    raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    medians = (
        pd.read_csv("models/impute_values.csv").set_index("column")["median"].to_dict()
    )
    clean, _ = clean_shipments(raw, medians=medians)
    df = build_features(clean, hubs)
    order = pd.read_csv("models/feature_order.csv").feature.tolist()
    if order != FEATURES:
        raise ValueError("Saved feature order differs from the pinned specification")
    pipe = joblib.load("models/delay_model.pkl")
    threshold = float(pd.read_csv("models/delay_threshold.csv").threshold.iloc[0])
    df["delay_probability"] = pipe.predict_proba(df[order])[:, 1].round(4)
    # The contract defines flagging against the exported four-decimal probability.
    df["flagged"] = df.delay_probability.ge(threshold).astype(int)
    segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]
    df = df.merge(segments, on="hub_id", how="left", sort=False, validate="many_to_one")
    if df.segment.isna().any():
        raise ValueError("Missing hub segment")
    df["action"] = "No action"
    df.loc[df.flagged.eq(1), "action"] = "Proactive customer call"
    df.loc[df.flagged.eq(1) & df.segment.eq("Escalate"), "action"] = (
        "Escalate to hub manager"
    )
    result = df[
        ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    ]
    assert (
        len(result) == 48
        and result.shipment_id.is_unique
        and not result.isna().any().any()
    )
    result.to_csv("outputs/new_shipment_actions.csv", index=False, float_format="%.4f")
    print(
        "48 shipments scored; flagged:",
        int(result.flagged.sum()),
        "manager escalations:",
        int(result.action.eq("Escalate to hub manager").sum()),
    )


if __name__ == "__main__":
    main()
