"""Profile hubs and compare K-Means segmentation at K=2,3,4,5."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("data/modelling_table.csv")
    profile = (
        df.groupby("hub_id", sort=True)
        .agg(
            shipments=("shipment_id", "size"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
    )
    profile.to_csv("outputs/hub_profile.csv", index=False, float_format="%.4f")
    columns = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaled = StandardScaler().fit_transform(profile[columns])
    # Only the upper triangle: all 276 unordered pairs, each counted once.
    i, j = np.triu_indices(len(profile), k=1)
    distances = np.linalg.norm(scaled[i] - scaled[j], axis=1)
    rows = []
    labels3 = None
    for k in [2, 3, 4, 5]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        same = labels[i] == labels[j]
        rows.append(
            dict(
                K=k,
                across_over_within=distances[~same].mean() / distances[same].mean(),
                cluster_sizes="-".join(
                    map(str, sorted(np.bincount(labels), reverse=True))
                ),
            )
        )
        if k == 3:
            labels3 = labels
    pd.DataFrame(rows).to_csv(
        "outputs/part_d_separation.csv", index=False, float_format="%.4f"
    )
    profile["cluster"] = labels3
    ordered = profile.groupby("cluster").delay_rate.mean().sort_values().index
    names = dict(zip(ordered, ["Standard", "Watch", "Escalate"]))
    profile["segment"] = profile.cluster.map(names)
    profile["segment"] = pd.Categorical(
        profile.segment, categories=["Standard", "Watch", "Escalate"], ordered=True
    )
    segment_profile = (
        profile.groupby("segment", observed=True)
        .agg(hubs=("hub_id", "size"), **{col: (col, "mean") for col in columns})
        .reset_index()
    )
    segment_profile.to_csv(
        "outputs/segment_profile.csv", index=False, float_format="%.4f"
    )
    segments = profile[["hub_id", "segment"]].merge(
        df[["hub_id", "hub_tier"]].drop_duplicates(),
        how="left",
        on="hub_id",
        validate="one_to_one",
    )
    segments.to_csv("outputs/hub_segments.csv", index=False)
    expected = segments.hub_tier.map({1: "Standard", 2: "Watch", 3: "Escalate"})
    mismatches = segments.loc[segments.segment.astype(str) != expected]
    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
    assert set(mismatches.hub_id) == {"HUB07", "HUB13", "HUB15", "HUB22"}
    print("Part D segments complete; four tier-versus-segment mismatches.")


if __name__ == "__main__":
    main()
