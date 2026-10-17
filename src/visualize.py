"""All charts. Every function saves a PNG into images/ (used as GitHub proof)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from . import config

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    config.IMG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.IMG / name, dpi=150)
    plt.close(fig)


def plot_sales_trend(df):
    daily = df.groupby("date")["qty_clean"].sum()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily.index, daily.values, alpha=0.35, label="Daily units")
    ax.plot(daily.index, daily.rolling(28).mean(), lw=2.2, label="28-day moving average")
    ax.set(title="Total Daily Sales - Trend & Seasonality", xlabel="Date", ylabel="Units sold")
    ax.legend()
    _save(fig, "02_sales_trend.png")


def plot_category_sales(df):
    rev = (df.assign(revenue=df["qty_sold"] * df["price"]).groupby("category")["revenue"].sum()
             .sort_values() / 1e6)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rev.plot.barh(ax=ax, color=sns.color_palette("viridis", len(rev)))
    ax.set(title="Revenue by Category (INR millions)", xlabel="Revenue (M)", ylabel="")
    _save(fig, "03_category_sales.png")


def plot_seasonality(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    dow = df.groupby(df["date"].dt.dayofweek)["qty_clean"].mean()
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow.plot.bar(ax=axes[0], color="#4c72b0", rot=0)
    axes[0].set(title="Average Units by Day of Week", xlabel="", ylabel="Units / series / day")
    mon = df.groupby(df["date"].dt.month)["qty_clean"].mean()
    axes[1].plot(mon.index, mon.values, marker="o")
    axes[1].set(title="Average Units by Month", xlabel="Month", ylabel="Units / series / day")
    _save(fig, "04_seasonality.png")


def plot_intermittency(p_zero):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.histplot(p_zero, bins=15, ax=ax, color="#dd8452")
    ax.axvline(config.INTERMITTENT_THRESHOLD, color="red", ls="--", label="Intermittent threshold")
    ax.set(title="Share of Zero-Demand Days per Series (P0)", xlabel="P0")
    ax.legend()
    _save(fig, "05_intermittency.png")


def plot_promo_lift(lift):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([f"{int(d*100)}%" for d in lift.index], lift.values, color="#55a868")
    ax.axhline(1, color="grey", ls="--")
    ax.set(title="Promotion Lift (demand vs series average)", xlabel="Discount", ylabel="Relative demand")
    _save(fig, "06_promo_lift.png")


def plot_forecast_vs_actual(bt, series_ids, fname="07_actual_vs_predicted.png"):
    fig, axes = plt.subplots(len(series_ids), 1, figsize=(11, 3.6 * len(series_ids)), squeeze=False)
    for ax, sid in zip(axes[:, 0], series_ids):
        g = bt[bt["series_id"] == sid].sort_values("date")
        ax.plot(g["date"], g["actual"], "k-o", ms=3, label="Actual")
        ax.plot(g["date"], g["hybrid"], lw=2, label="Hybrid forecast")
        ax.plot(g["date"], g["snaive"], ls=":", label="Seasonal naive")
        ax.set(title=f"Backtest (last 28 days) - {sid}", ylabel="Units")
        ax.legend(ncol=3)
    _save(fig, fname)


def plot_model_comparison(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    summary["MAE"].plot.bar(ax=axes[0], color="#4c72b0", rot=0)
    axes[0].set_title("MAE by method (lower = better)")
    summary["WAPE_%"].plot.bar(ax=axes[1], color="#c44e52", rot=0)
    axes[1].set_title("WAPE % by method (lower = better)")
    _save(fig, "08_model_comparison.png")


def plot_future_forecast(df, fc, series_id):
    h = df[df["series_id"] == series_id].sort_values("date").tail(90)
    f = fc[fc["series_id"] == series_id].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(h["date"], h["qty_clean"], label="History (last 90 days)")
    ax.plot(f["date"], f["forecast"], lw=2.4, color="crimson", label="28-day forecast")
    ax.set(title=f"Demand Forecast - {series_id}", ylabel="Units")
    ax.legend()
    _save(fig, "09_future_forecast.png")


def plot_inventory_status(rec):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {"CRITICAL": "#c44e52", "REORDER": "#dd8452", "OK": "#55a868", "OVERSTOCK": "#8172b3"}
    counts = rec["status"].value_counts()
    ax.bar(counts.index, counts.values, color=[colors[s] for s in counts.index])
    ax.set(title="Inventory Status across 36 Store-Item Pairs", ylabel="# SKU-store")
    _save(fig, "10_inventory_status.png")


def plot_top_reorders(rec, n=10):
    top = rec[rec["order_qty"] > 0].nlargest(n, "order_value").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["store_id"] + " " + top["item_name"], top["order_value"] / 1e3, color="#4c72b0")
    ax.set(title="Top Recommended Purchase Orders (INR thousands)", xlabel="Order value (K)")
    _save(fig, "11_top_reorders.png")


def plot_kpi(summ):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    summ["fill_rate"].mul(100).plot.bar(ax=axes[0], color=["#c44e52", "#55a868"], rot=0)
    axes[0].set(title="Average Fill Rate (%)", ylim=(max(0, summ['fill_rate'].min() * 100 - 5), 100.5))
    summ["stockout_days"].plot.bar(ax=axes[1], color=["#c44e52", "#55a868"], rot=0)
    axes[1].set_title("Total Stock-out Days")
    (summ["avg_inventory_value"] / 1e3).plot.bar(ax=axes[2], color=["#c44e52", "#55a868"], rot=0)
    axes[2].set_title("Average Inventory Value (INR K)")
    _save(fig, "12_kpi_comparison.png")
