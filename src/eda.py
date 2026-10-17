"""Exploratory analysis helpers."""
import pandas as pd


def intermittency_table(df):
    """Share of zero-demand days (P0) per series. High P0 => Croston/SBA candidates."""
    return (df.assign(z=(df["qty_clean"] == 0))
              .groupby("series_id")["z"].mean().rename("p_zero"))


def promo_lift(df):
    """Average demand uplift (vs the series' own mean) by discount level."""
    rel = df["qty_clean"] / df.groupby("series_id")["qty_clean"].transform("mean").replace(0, pd.NA)
    return rel.astype(float).groupby(df["discount_pct"]).mean().rename("relative_demand")
