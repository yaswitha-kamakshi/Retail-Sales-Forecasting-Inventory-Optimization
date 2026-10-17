import numpy as np
from src.inventory import inventory_policy, simulate_policy
from src.models import croston_sba, seasonal_naive


def test_croston_constant_demand():
    y = np.array([0, 4, 0, 0, 4, 0, 0, 4] * 10)
    f = croston_sba(y, h=5, sba=False)
    assert f.shape == (5,) and 1.0 < f[0] < 2.0     # ~ 4 units every 3 days = 1.33/day


def test_croston_all_zero():
    assert croston_sba(np.zeros(30), h=3).sum() == 0


def test_seasonal_naive_repeats_week():
    y = np.tile(np.arange(7), 8).astype(float)
    assert np.allclose(seasonal_naive(y, 7), np.arange(7))


def test_inventory_policy_formulas():
    p = inventory_policy([10] * 28, resid_std=4, on_hand=50, lead_time=4, annual_demand=3650,
                         ordering_cost=500, unit_cost=100, holding_rate=0.2, service=0.95)
    assert p["mu_L"] == 40
    assert abs(p["safety_stock"] - 1.645 * 4 * 2) < 0.05          # z * sigma * sqrt(L)
    assert abs(p["eoq"] - np.sqrt(2 * 3650 * 500 / 20)) < 1e-6
    assert p["status"] in {"REORDER", "CRITICAL"} and p["order_qty"] >= p["eoq"]


def test_no_order_when_stock_high():
    p = inventory_policy([10] * 28, 2, on_hand=500, lead_time=3, annual_demand=3650,
                         ordering_cost=500, unit_cost=100, holding_rate=0.2)
    assert p["order_qty"] == 0


def test_simulation_fill_rate_bounds():
    r = simulate_policy(np.full(30, 5.0), start_stock=50, rop=20, q=60, lead=3)
    assert 0 <= r["fill_rate"] <= 1
