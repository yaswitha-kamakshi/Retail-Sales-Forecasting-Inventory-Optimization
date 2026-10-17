"""Inventory optimisation: safety stock, reorder point, EOQ, order quantity, KPI simulation."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from . import config


def inventory_policy(forecast, resid_std, on_hand, lead_time, annual_demand,
                     ordering_cost, unit_cost, holding_rate, service=config.SERVICE_LEVEL, on_order=0):
    """Compute SS / ROP / EOQ / order qty for ONE store-item.

    mu_L    = forecast demand during lead time
    sigma_L = daily forecast-error std * sqrt(L)
    SS      = z * sigma_L          ROP = mu_L + SS
    EOQ     = sqrt(2*D*K / H),  H = unit_cost * holding_rate
    Order is triggered when inventory position (on-hand + on-order) <= ROP.
    """
    forecast = np.asarray(forecast, float)
    L = int(lead_time)
    z = norm.ppf(service)
    mu_L = forecast[:L].sum()
    sigma_L = resid_std * np.sqrt(L)
    ss = z * sigma_L
    rop = mu_L + ss
    h = unit_cost * holding_rate
    eoq = np.sqrt(2 * annual_demand * ordering_cost / h) if h > 0 else mu_L
    position = on_hand + on_order          # inventory position = on-hand + already ordered
    qty = max(eoq, rop - position) if position <= rop else 0.0
    daily = forecast.mean()
    cover = on_hand / daily if daily > 0 else np.inf
    if position < mu_L:
        status = "CRITICAL"       # even with open orders, stock-out likely within lead time
    elif position <= rop:
        status = "REORDER"
    elif position > rop + eoq:
        status = "OVERSTOCK"
    else:
        status = "OK"
    return dict(mu_L=mu_L, sigma_L=sigma_L, safety_stock=ss, reorder_point=rop, eoq=eoq,
                order_qty=int(np.ceil(qty)), days_of_cover=cover, status=status)


def build_recommendations(fc, summary, service=config.SERVICE_LEVEL):
    """fc: columns series_id, date, forecast.  summary: one row per series (see main.py)."""
    rows = []
    for _, s in summary.iterrows():
        f = fc[fc["series_id"] == s["series_id"]].sort_values("date")["forecast"].values
        p = inventory_policy(f, s["resid_std"], s["on_hand"], s["lead_time_days"], s["annual_demand"],
                             s["ordering_cost"], s["unit_cost"], s["holding_cost_rate"], service,
                             on_order=s.get("on_order", 0))
        rows.append({**{k: s[k] for k in ["store_id", "city", "item_id", "item_name", "category"]},
                     "on_hand": s["on_hand"], "on_order": s.get("on_order", 0), "lead_time_days": s["lead_time_days"],
                     "forecast_28d": round(f.sum(), 1), **p,
                     "order_value": p["order_qty"] * s["unit_cost"]})
    rec = pd.DataFrame(rows)
    num = ["mu_L", "sigma_L", "safety_stock", "reorder_point", "eoq", "days_of_cover"]
    rec[num] = rec[num].round(1)
    order = {"CRITICAL": 0, "REORDER": 1, "OVERSTOCK": 2, "OK": 3}
    return rec.sort_values(by="status", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)


def simulate_policy(demand, start_stock, rop, q, lead):
    """Day-by-day (s, Q) simulation. Returns fill rate, stock-out days, average inventory."""
    on_hand, pipe, sold, so_days, inv = start_stock, {}, 0.0, 0, []
    for t, d in enumerate(demand):
        on_hand += pipe.pop(t, 0)
        s = min(d, on_hand)
        on_hand -= s
        sold += s
        so_days += int(d > s)
        if on_hand + sum(pipe.values()) <= rop:
            pipe[t + lead] = pipe.get(t + lead, 0) + q
        inv.append(on_hand)
    return dict(fill_rate=sold / max(demand.sum(), 1e-9), stockout_days=so_days,
                avg_inventory=float(np.mean(inv)))


def kpi_comparison(df, summary, service=config.SERVICE_LEVEL, days=config.KPI_SIM_DAYS):
    """Replay the last `days` days (parameters learned from data BEFORE them):
    Policy A 'naive'      : ROP = average lead-time demand (no safety stock)
    Policy B 'optimised'  : ROP = average lead-time demand + z * sigma * sqrt(L)
    Both use the same EOQ order quantity and the same starting stock."""
    z, cutoff, rows = norm.ppf(service), df["date"].max() - pd.Timedelta(days=days), []
    for _, s in summary.iterrows():
        g = df[df["series_id"] == s["series_id"]].sort_values("date")
        hist, future = g[g["date"] <= cutoff]["qty_clean"], g[g["date"] > cutoff]["qty_clean"].values
        mu_d, sd = hist.tail(28).mean(), hist.tail(56).std()
        L = int(s["lead_time_days"])
        h = s["unit_cost"] * s["holding_cost_rate"]
        q = max(1, int(np.sqrt(2 * mu_d * 365 * s["ordering_cost"] / h)))
        rop_a, rop_b = mu_d * L, mu_d * L + z * sd * np.sqrt(L)
        start = rop_b + q / 2
        for name, rop in [("naive", rop_a), ("optimised", rop_b)]:
            r = simulate_policy(future, start, rop, q, L)
            rows.append(dict(series_id=s["series_id"], policy=name, **r,
                             inventory_value=r["avg_inventory"] * s["unit_cost"]))
    kpi = pd.DataFrame(rows)
    summ = kpi.groupby("policy").agg(fill_rate=("fill_rate", "mean"),
                                     stockout_days=("stockout_days", "sum"),
                                     avg_inventory_value=("inventory_value", "sum")).round(4)
    return kpi, summ
