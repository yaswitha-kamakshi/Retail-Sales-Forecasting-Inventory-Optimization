"""Run the full pipeline:  python main.py   (about 1-2 minutes)

generate data -> clean -> EDA -> features -> backtest -> final forecast -> inventory policy -> KPIs
"""
import argparse
import joblib
import numpy as np
import pandas as pd

from src import config, data_generator, preprocessing, eda, models, inventory, visualize as viz
from src.features import make_encoders, build_training_features

pd.set_option("display.width", 200, "display.max_columns", 30)


def main(regenerate=False):
    config.OUT.mkdir(exist_ok=True); config.MODELS.mkdir(exist_ok=True)

    # 1. DATA ------------------------------------------------------------
    if regenerate or not config.DATA_RAW.exists():
        print("[1/7] Generating synthetic retail dataset ...")
        data_generator.generate()
    raw = preprocessing.load_raw(config.DATA_RAW)
    print("[1/7] Data quality report:")
    for k, v in preprocessing.quality_report(raw).items():
        print(f"      {k:>22}: {v}")

    # 2. CLEAN -----------------------------------------------------------
    df = preprocessing.clean(raw)
    df.to_csv(config.OUT / "cleaned_data.csv", index=False)
    print(f"[2/7] Cleaned data: {len(df):,} rows. Stock-out days repaired: {int((raw.stockout_flag==1).sum())}")

    # 3. EDA -------------------------------------------------------------
    p_zero = eda.intermittency_table(df)
    p_zero.to_csv(config.OUT / "intermittency.csv")
    viz.plot_sales_trend(df); viz.plot_category_sales(df); viz.plot_seasonality(df)
    viz.plot_intermittency(p_zero); viz.plot_promo_lift(eda.promo_lift(df))
    n_int = int((p_zero > config.INTERMITTENT_THRESHOLD).sum())
    print(f"[3/7] EDA charts saved to images/. Intermittent series (P0>{config.INTERMITTENT_THRESHOLD}): {n_int}/{len(p_zero)}")

    # 4. BACKTEST --------------------------------------------------------
    enc = make_encoders(df)
    print("[4/7] Backtesting on the last 28 days (train on the past only) ...")
    bt, train = models.backtest(df, enc, p_zero)
    per_series = models.compute_metrics(bt, train)
    summary = models.summarise(per_series)
    meta = df.drop_duplicates("series_id").set_index("series_id")[["category"]]
    per_series = per_series.join(meta, on="series_id")
    per_series["segment"] = np.where(per_series["series_id"].map(p_zero) > config.INTERMITTENT_THRESHOLD,
                                     "intermittent", "regular")
    seg = per_series[per_series.method == "hybrid"].groupby("segment")[["mae", "mase"]].mean().round(3)
    cat = per_series[per_series.method == "hybrid"].groupby("category")[["mae", "mase"]].mean().round(3)
    summary.to_csv(config.OUT / "backtest_metrics.csv")
    per_series.to_csv(config.OUT / "backtest_metrics_by_series.csv", index=False)
    bt.to_csv(config.OUT / "backtest_predictions.csv", index=False)
    print(summary, "\n\nHybrid accuracy by segment:\n", seg, "\n\nHybrid accuracy by category:\n", cat)
    best_reg = per_series[(per_series.method == "hybrid") & (per_series.segment == "regular")].nsmallest(1, "mase")["series_id"].iloc[0]
    best_int = per_series[(per_series.segment == "intermittent")]["series_id"].iloc[0]
    viz.plot_forecast_vs_actual(bt, [best_reg, best_int]); viz.plot_model_comparison(summary)

    # 5. FINAL FORECAST (train on ALL data) -------------------------------
    print("[5/7] Training final model on all data and forecasting the next 28 days ...")
    final_fc, model = models.make_forecasts(df, config.HORIZON, enc, p_zero)
    joblib.dump({"model": model, "encoders": enc}, config.MODELS / "rf_model.pkl")
    fc = final_fc.rename(columns={"hybrid": "forecast"})[["series_id", "store_id", "item_id", "date", "forecast", "rf", "snaive", "sba"]]
    fc.to_csv(config.OUT / "forecast_next_28d.csv", index=False)
    viz.plot_future_forecast(df, fc, best_reg)

    # 6. INVENTORY OPTIMISATION -----------------------------------------
    last = df.sort_values("date").groupby("series_id").tail(1).set_index("series_id")
    resid = per_series[per_series.method == "hybrid"].set_index("series_id")["rmse"].clip(lower=0.5)
    sm = last[["store_id", "city", "item_id", "item_name", "category", "stock_on_hand", "on_order", "lead_time_days",
               "unit_cost", "ordering_cost", "holding_cost_rate"]].rename(columns={"stock_on_hand": "on_hand"})
    sm["annual_demand"] = df[df["date"] > df["date"].max() - pd.Timedelta(days=90)].groupby("series_id")["qty_clean"].mean() * 365
    sm["resid_std"] = resid
    sm = sm.reset_index()
    sm.to_csv(config.OUT / "series_summary.csv", index=False)
    rec = inventory.build_recommendations(fc, sm, config.SERVICE_LEVEL)
    rec.to_csv(config.OUT / "inventory_recommendations.csv", index=False)
    viz.plot_inventory_status(rec); viz.plot_top_reorders(rec)
    print("[6/7] Inventory recommendations (top 12 by urgency):")
    cols = ["store_id", "item_name", "on_hand", "on_order", "lead_time_days", "mu_L", "safety_stock", "reorder_point", "eoq", "order_qty", "status"]
    print(rec[cols].head(12).to_string(index=False))
    print("\n      Status counts:", rec["status"].value_counts().to_dict(),
          "| Total PO value: INR", f"{rec['order_value'].sum():,.0f}")

    # 7. KPI SIMULATION ---------------------------------------------------
    kpi, ksum = inventory.kpi_comparison(df, sm)
    kpi.to_csv(config.OUT / "kpi_simulation_by_series.csv", index=False)
    ksum.to_csv(config.OUT / "kpi_comparison.csv")
    viz.plot_kpi(ksum)
    print("[7/7] Inventory policy replay over last", config.KPI_SIM_DAYS, "days:\n", ksum)
    print("\nDone. Outputs -> outputs/, charts -> images/, model -> models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="regenerate the synthetic dataset")
    main(ap.parse_args().regenerate)
