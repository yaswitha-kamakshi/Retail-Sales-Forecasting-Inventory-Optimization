"""Loading, quality checks and cleaning (incl. stock-out censoring correction)."""
import numpy as np
import pandas as pd

KEY = ["store_id", "item_id", "date"]


def load_raw(path):
    df = pd.read_csv(path, parse_dates=["date"])
    assert {"store_id", "item_id", "date", "qty_sold"}.issubset(df.columns), "Missing required columns"
    return df


def quality_report(df):
    """Return a dict of data-quality metrics (printed in main.py)."""
    n_days = df.groupby(["store_id", "item_id"])["date"].nunique()
    span = (df["date"].max() - df["date"].min()).days + 1
    so = df["stockout_flag"] if "stockout_flag" in df else pd.Series([0])
    return {
        "rows": len(df),
        "series (store x item)": int(df.groupby(["store_id", "item_id"]).ngroups),
        "date_range": f"{df['date'].min().date()} -> {df['date'].max().date()}",
        "duplicates": int(df.duplicated(KEY).sum()),
        "missing_qty": int(df["qty_sold"].isna().sum()),
        "negative_qty": int((df["qty_sold"] < 0).sum()),
        "missing_days": int((span - n_days).sum()),
        "stockout_days": int(so.sum()),
        "stockout_rate_%": round(100 * so.mean(), 2),
    }


def clean(df):
    """Clean data and build `qty_clean` = demand estimate corrected for stock-outs.

    Sales on stock-out days are CENSORED (we sold everything we had, true demand was higher).
    Instead of dropping those rows (which breaks the time series), we replace them by
    max(observed sales, interpolated neighbours).
    """
    df = df.drop_duplicates(KEY).copy()
    df.loc[df["qty_sold"] < 0, "qty_sold"] = np.nan
    for c in ["on_promo", "discount_pct"]:
        if c in df:
            df[c] = df[c].fillna(0)
    df["price"] = df.groupby("item_id")["price"].transform(lambda s: s.fillna(s.max()))
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    df["series_id"] = df["store_id"] + "_" + df["item_id"]

    q = df["qty_sold"].astype(float)
    so = df["stockout_flag"] == 1 if "stockout_flag" in df else pd.Series(False, index=df.index)
    censored = q.isna() | so
    q[censored] = np.nan
    interp = q.groupby(df["series_id"]).transform(lambda s: s.interpolate(limit_direction="both"))
    df["qty_clean"] = np.where(censored, np.maximum(interp, df["qty_sold"].fillna(0)), q)
    df["qty_clean"] = df["qty_clean"].fillna(0)
    return df
