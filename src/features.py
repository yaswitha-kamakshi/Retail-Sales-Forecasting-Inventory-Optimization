"""Feature engineering: lags, rolling stats, calendar, promo, encodings.

Two code paths share the same definitions:
  * build_training_features -> vectorised, for training
  * row_features            -> one row, for recursive multi-step forecasting
"""
import numpy as np
import pandas as pd
from .calendar_utils import holiday_flag

LAGS = (1, 7, 14)
WINDOWS = (7, 14, 28)
FEATURE_COLS = (
    [f"lag_{l}" for l in LAGS]
    + [f"rollmean_{w}" for w in WINDOWS] + [f"rollstd_{w}" for w in WINDOWS]
    + ["dow", "week", "month", "is_weekend", "holiday_flag",
       "on_promo", "discount_pct", "price", "store_code", "item_code", "category_code"]
)


def make_encoders(df):
    return {c: {v: i for i, v in enumerate(sorted(df[c].unique()))}
            for c in ["store_id", "item_id", "category"]}


def build_training_features(df, enc):
    df = df.sort_values(["series_id", "date"]).copy()
    g = df.groupby("series_id")["qty_clean"]
    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l)
    for w in WINDOWS:  # shift(1) => only PAST data, no leakage
        df[f"rollmean_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
        df[f"rollstd_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w).std())
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["holiday_flag"] = holiday_flag(df["date"])
    df["store_code"] = df["store_id"].map(enc["store_id"])
    df["item_code"] = df["item_id"].map(enc["item_id"])
    df["category_code"] = df["category"].map(enc["category"])
    return df.dropna(subset=FEATURE_COLS)  # drop cold-start rows


def row_features(hist, date, static):
    """Features for ONE future day from the running history array `hist`."""
    r = {f"lag_{l}": hist[-l] for l in LAGS}
    for w in WINDOWS:
        r[f"rollmean_{w}"] = hist[-w:].mean()
        r[f"rollstd_{w}"] = hist[-w:].std(ddof=1)
    r.update(dow=date.dayofweek, week=int(date.isocalendar()[1]), month=date.month,
             is_weekend=int(date.dayofweek >= 5), holiday_flag=int(holiday_flag([date])[0]),
             on_promo=0, discount_pct=0.0,   # future promotions unknown -> assume none
             price=static["price"], store_code=static["store_code"],
             item_code=static["item_code"], category_code=static["category_code"])
    return r
