"""Planner dashboard.  Run:  streamlit run app/streamlit_app.py   (after `python main.py`)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import streamlit as st
from src import config, inventory

st.set_page_config(page_title="Retail Forecast & Replenishment", layout="wide")
st.title("Retail Sales Forecasting & Inventory Optimization")


@st.cache_data
def load():
    hist = pd.read_csv(config.OUT / "cleaned_data.csv", parse_dates=["date"])
    fc = pd.read_csv(config.OUT / "forecast_next_28d.csv", parse_dates=["date"])
    sm = pd.read_csv(config.OUT / "series_summary.csv")
    rec = pd.read_csv(config.OUT / "inventory_recommendations.csv")
    kpi = pd.read_csv(config.OUT / "kpi_comparison.csv", index_col=0)
    return hist, fc, sm, rec, kpi


try:
    hist, fc, sm, rec, kpi = load()
except FileNotFoundError:
    st.error("Run `python main.py` first to create the outputs.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Forecast & replenishment", "Network overview", "Policy KPIs"])

with tab1:
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", sorted(sm["store_id"].unique()))
    items = sm[sm["store_id"] == store]
    item = c2.selectbox("Item", items["item_id"] + " - " + items["item_name"])
    row = items[items["item_id"] == item.split(" - ")[0]].iloc[0]
    sid = row["series_id"]

    c3, c3b, c4, c5 = st.columns(4)
    on_hand = c3.number_input("On-hand stock", 0, 100000, int(row["on_hand"]))
    on_order = c3b.number_input("On-order (in transit)", 0, 100000, int(row["on_order"]))
    lead = c4.number_input("Lead time (days)", 1, 28, int(row["lead_time_days"]))
    service = c5.slider("Service level", 0.80, 0.995, config.SERVICE_LEVEL, 0.005)

    h = hist[hist["series_id"] == sid].tail(90)
    f = fc[fc["series_id"] == sid].sort_values("date")
    st.line_chart(pd.concat([h.set_index("date")["qty_clean"].rename("history"),
                             f.set_index("date")["forecast"].rename("forecast")], axis=1))

    p = inventory.inventory_policy(f["forecast"].values, row["resid_std"], on_hand, lead, row["annual_demand"],
                                   row["ordering_cost"], row["unit_cost"], row["holding_cost_rate"], service, on_order=on_order)
    m = st.columns(5)
    m[0].metric("Lead-time demand", f"{p['mu_L']:.0f}")
    m[1].metric("Safety stock", f"{p['safety_stock']:.0f}")
    m[2].metric("Reorder point", f"{p['reorder_point']:.0f}")
    m[3].metric("EOQ", f"{p['eoq']:.0f}")
    m[4].metric("Order now", f"{p['order_qty']}")
    {"CRITICAL": st.error, "REORDER": st.warning, "OVERSTOCK": st.info, "OK": st.success}[p["status"]](
        f"Status: {p['status']} | Days of cover: {p['days_of_cover']:.1f}")

with tab2:
    st.dataframe(rec, use_container_width=True)
    st.download_button("Download PO recommendations (CSV)", rec.to_csv(index=False), "recommendations.csv")
    st.bar_chart(rec["status"].value_counts())

with tab3:
    st.caption("Replay of the last 84 days: naive ROP (no safety stock) vs optimised ROP (with safety stock).")
    st.dataframe(kpi)
