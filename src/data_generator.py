"""Synthetic retail dataset generator (virtual simulation of a retail chain).

Simulates, per store x item x day:
  * base demand x store size x day-of-week x yearly seasonality x growth trend
  * festive-season / summer uplifts, weekly promotions with discount lift
  * Poisson randomness (so low-volume items become *intermittent*)
  * a real inventory process with lead times -> genuine stock-outs that CENSOR sales
"""
import numpy as np
import pandas as pd
from . import config
from .calendar_utils import holiday_flag

STORES = [("S01", "Visakhapatnam", 1.0), ("S02", "Hyderabad", 1.4), ("S03", "Bengaluru", 1.2)]

# item_id, name, category, base daily demand, list price, unit cost, lead time (days), shelf life
ITEMS = [
    ("I001", "Basmati Rice 5kg",    "Grocery",       14, 520, 410, 5, 365),
    ("I002", "Sunflower Oil 1L",    "Grocery",       18, 160, 130, 4, 270),
    ("I003", "Toor Dal 1kg",        "Grocery",       12, 165, 135, 5, 300),
    ("I004", "Cola 2L",             "Beverages",     22,  95,  70, 3, 180),
    ("I005", "Orange Juice 1L",     "Beverages",      9, 120,  85, 3,  90),
    ("I006", "Potato Chips",        "Snacks",        25,  20,  14, 3, 120),
    ("I007", "Chocolate Gift Box",  "Snacks",         7, 350, 250, 6, 200),
    ("I008", "Shampoo 340ml",       "Personal Care",  8, 280, 190, 7, 700),
    ("I009", "Face Wash 100ml",     "Personal Care",  1.0, 210, 140, 7, 700),   # intermittent
    ("I010", "Detergent 2kg",       "Household",     10, 380, 290, 6, 720),
    ("I011", "LED Bulb 9W",         "Household",      1.1, 110,  70, 8, 900),   # intermittent
    ("I012", "Olive Oil 500ml",     "Grocery",        0.8, 640, 480, 9, 540),   # intermittent
]
DOW = np.array([0.90, 0.85, 0.90, 0.95, 1.10, 1.30, 1.25])  # Mon..Sun


def generate(seed=config.SEED, n_stores=3, n_items=12, save=True):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(config.START, config.END)
    n = len(dates)
    t = np.arange(n)
    doy, dow, month = dates.dayofyear.values, dates.dayofweek.values, dates.month.values
    hol = holiday_flag(dates)
    weeks = t // 7
    frames = []

    for sid, city, store_f in STORES[:n_stores]:
        for iid, name, cat, base, price, cost, lead, shelf in ITEMS[:n_items]:
            # --- promotions: whole-week promos with 10/15/20% discount ---
            promo_week = rng.random(weeks.max() + 1) < 0.12
            disc_week = rng.choice([0.10, 0.15, 0.20], size=weeks.max() + 1)
            disc = np.where(promo_week[weeks], disc_week[weeks], 0.0)

            # --- demand components ---
            season = 1 + 0.15 * np.sin(2 * np.pi * (doy - 60) / 365)
            trend = 1 + 0.12 * t / 365
            festive = 1 + 0.30 * hol + (0.80 * hol if iid == "I007" else 0)
            summer = 1 + (0.30 * np.isin(month, [4, 5, 6]) if cat == "Beverages" else 0)
            lam = (base * store_f * DOW[dow] * season * trend * festive * summer
                   * (1 + 2.5 * disc) * rng.lognormal(0, 0.15, n))
            demand = rng.poisson(lam)

            # --- inventory simulation with (s, S) replenishment -> censored sales ---
            mean_d = base * store_f
            rop, order_up_to = mean_d * (lead + 2), mean_d * (lead + 10)
            on_hand, pipe = order_up_to, {}
            sold, stock_end, so_flag = np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
            on_order = np.zeros(n)
            for i in range(n):
                on_hand += pipe.pop(i, 0)
                s = min(demand[i], on_hand)
                so_flag[i] = int(demand[i] > on_hand)
                on_hand -= s
                sold[i], stock_end[i] = s, on_hand
                position = on_hand + sum(pipe.values())
                if position <= rop:
                    pipe[i + lead] = pipe.get(i + lead, 0) + int(np.ceil(order_up_to - position))
                on_order[i] = sum(pipe.values())

            frames.append(pd.DataFrame({
                "date": dates, "store_id": sid, "city": city, "item_id": iid,
                "item_name": name, "category": cat, "qty_sold": sold.astype(int),
                "price": np.round(price * (1 - disc), 2), "discount_pct": disc,
                "on_promo": (disc > 0).astype(int), "stock_on_hand": stock_end.astype(int),
                "on_order": on_order.astype(int),
                "stockout_flag": so_flag, "unit_cost": cost, "lead_time_days": lead,
                "holding_cost_rate": 0.20, "ordering_cost": 500, "shelf_life_days": shelf,
            }))

    df = pd.concat(frames, ignore_index=True)
    if save:
        config.DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.DATA_RAW, index=False)
    return df
