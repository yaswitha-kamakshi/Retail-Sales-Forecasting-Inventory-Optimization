"""Forecasting models: seasonal naive, Croston-SBA, Random Forest, hybrid + backtesting."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from . import config
from .features import FEATURE_COLS, build_training_features, row_features

METHODS = ["snaive", "sba", "rf", "hybrid"]


# ---------- statistical baselines ----------
def seasonal_naive(y, h, season=7, k=4):
    """Average of the last k occurrences of the same weekday."""
    y = np.asarray(y, float)
    out = []
    for i in range(h):
        idx = len(y) - season + (i % season)
        out.append(np.mean([y[idx - season * j] for j in range(k) if idx - season * j >= 0]))
    return np.array(out)


def croston_sba(y, alpha=0.1, h=28, sba=True):
    """Croston's method for intermittent demand; SBA = Syntetos-Boylan bias correction."""
    y = np.asarray(y, float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return np.zeros(h)
    z = y[nz]                       # non-zero demand sizes
    p = np.diff(np.r_[-1, nz])      # inter-demand intervals
    z_hat, p_hat = z[0], p[0]
    for zi, pi in zip(z[1:], p[1:]):
        z_hat += alpha * (zi - z_hat)
        p_hat += alpha * (pi - p_hat)
    f = z_hat / p_hat
    if sba:
        f *= (1 - alpha / 2)
    return np.full(h, f)


# ---------- machine learning ----------
def train_rf(feat_df):
    rf = RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=5,
                               n_jobs=-1, random_state=config.SEED)
    rf.fit(feat_df[FEATURE_COLS], feat_df["qty_clean"])
    return rf


def recursive_forecast(model, train, horizon, enc):
    """Multi-step forecast: each predicted day is fed back as history (all series in batch)."""
    state = {}
    for sid, g in train.sort_values("date").groupby("series_id"):
        last = g.iloc[-1]
        state[sid] = dict(
            hist=list(g["qty_clean"].values[-60:]), store_id=last["store_id"], item_id=last["item_id"],
            price=g["price"].tail(60).max(),
            store_code=enc["store_id"][last["store_id"]], item_code=enc["item_id"][last["item_id"]],
            category_code=enc["category"][last["category"]])
    start, out = train["date"].max(), []
    for h in range(1, horizon + 1):
        d = start + pd.Timedelta(days=h)
        X = pd.DataFrame([row_features(np.array(s["hist"]), d, s) for s in state.values()])[FEATURE_COLS]
        pred = np.clip(model.predict(X), 0, None)
        for (sid, s), p in zip(state.items(), pred):
            s["hist"].append(p)
            out.append(dict(series_id=sid, store_id=s["store_id"], item_id=s["item_id"], date=d, rf=p))
    return pd.DataFrame(out)


def make_forecasts(train, horizon, enc, p_zero, model=None):
    """Produce all methods + the hybrid forecast for every series."""
    if model is None:
        model = train_rf(build_training_features(train, enc))
    rf = recursive_forecast(model, train, horizon, enc)
    parts = []
    for sid, g in train.sort_values("date").groupby("series_id"):
        y = g["qty_clean"].values
        fr = rf[rf["series_id"] == sid].sort_values("date").copy()
        fr["snaive"] = seasonal_naive(y, horizon)
        fr["sba"] = croston_sba(y, h=horizon)
        blend = config.RF_WEIGHT * fr["rf"] + (1 - config.RF_WEIGHT) * fr["snaive"]
        fr["hybrid"] = fr["sba"] if p_zero[sid] > config.INTERMITTENT_THRESHOLD else blend
        parts.append(fr)
    return pd.concat(parts, ignore_index=True), model


# ---------- evaluation ----------
def backtest(df, enc, p_zero, horizon=config.HORIZON):
    """Hold out the last `horizon` days: train on the past only, forecast, compare."""
    cutoff = df["date"].max() - pd.Timedelta(days=horizon)
    train, test = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    fc, _ = make_forecasts(train, horizon, enc, p_zero)
    bt = fc.merge(test[["series_id", "date", "qty_clean"]].rename(columns={"qty_clean": "actual"}),
                  on=["series_id", "date"])
    return bt, train


def compute_metrics(bt, train):
    rows = []
    for sid, g in bt.groupby("series_id"):
        y = train[train["series_id"] == sid].sort_values("date")["qty_clean"].values
        scale = np.abs(y[7:] - y[:-7]).mean()          # in-sample seasonal-naive MAE
        for m in METHODS:
            e = g["actual"] - g[m]
            rows.append(dict(series_id=sid, method=m, mae=np.abs(e).mean(),
                             rmse=np.sqrt((e ** 2).mean()), bias=-e.mean(),
                             mase=np.abs(e).mean() / scale if scale > 0 else np.nan,
                             abs_err=np.abs(e).sum(), actual_sum=g["actual"].sum()))
    return pd.DataFrame(rows)


def summarise(per_series):
    s = per_series.groupby("method").agg(MAE=("mae", "mean"), RMSE=("rmse", "mean"),
                                         MASE=("mase", "mean"), abs_err=("abs_err", "sum"),
                                         actual=("actual_sum", "sum"))
    s["WAPE_%"] = 100 * s["abs_err"] / s["actual"]
    return s.drop(columns=["abs_err", "actual"]).loc[METHODS].round(3)
