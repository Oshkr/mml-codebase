import torch
import math
import pandas as pd

def load_gps_tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df[["lat", "lon"]].values, dtype=torch.float32)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geo_accuracy(retrieved_df: dict, thresholds_km=(1, 25, 200, 750, 2500)) -> dict:
    errors = [
        haversine_km(lt, ln, plt, pln)
        for lt, ln, plt, pln in zip(
            retrieved_df["lat"], retrieved_df["lon"],
            retrieved_df["pred_lat"], retrieved_df["pred_lon"],
        )
    ]
    return {f"Acc@{t}km": sum(e <= t for e in errors) / len(errors) * 100
            for t in thresholds_km}