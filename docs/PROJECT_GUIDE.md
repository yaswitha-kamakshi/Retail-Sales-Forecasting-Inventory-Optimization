# Project Guide — Retail Sales Forecasting & Inventory Optimization

Sections follow the A–P order from the brief. The **code (G)** is in `src/`, `main.py`, `app/`, `tests/` and the **README (K)** is in `README.md`.

---
## A. Project explanation

### Simple language
A shop must decide every day: *how many units of each product should I order, and when?* Order too little → customers find empty shelves (**stock-out**, lost sales). Order too much → money is stuck in unsold goods (**overstock**, storage cost, expiry). This system looks at past sales, predicts the next weeks of demand, and then calculates *when to reorder and how much*.

### Technical language
A two-stage pipeline: (1) a probabilistic-ish **demand forecaster** at store × SKU × day grain (global Random Forest on lag/rolling/calendar features, Croston-SBA for intermittent demand, seasonal-naive baseline, blended into a hybrid); (2) a **stochastic inventory policy** (continuous-review *(s, Q)*) where lead-time demand is `μ_L = Σ ŷ`, uncertainty `σ_L = σ_e·√L` (σ_e = backtest RMSE), safety stock `z·σ_L`, reorder point `μ_L + SS`, and `Q = EOQ`.

### Why it matters / who uses it
Supermarkets (D-Mart, Reliance Retail), quick-commerce and e-commerce (BigBasket, Amazon, Flipkart), Walmart: inventory is typically their largest asset. Better forecasts → higher **fill rate**, lower **inventory days**, less **wastage**, more **working capital** freed. Forecasting drives demand planning, stock planning, avoiding overstock, avoiding stock-outs, and ultimately profit.

### Workflow
Historical sales → cleaning & QC → EDA → feature engineering → forecasting → backtest → safety stock / ROP / EOQ → PO recommendations → dashboard & reports.

---
## B. Tech-stack options

| | Option A – Easiest | Option B – Intermediate (**chosen**) | Option C – Advanced |
|---|---|---|---|
| Tools | Pandas, Matplotlib, Jupyter | + scikit-learn, SciPy, Seaborn, Streamlit | + LightGBM/XGBoost, Prophet/statsmodels, MLflow, FastAPI, Docker |
| Models | Moving average, linear regression | Random Forest + Croston-SBA + seasonal naive hybrid | Per-series SARIMAX/Prophet, LightGBM quantile, ensembles, hierarchical reconciliation |
| Difficulty | Low | Medium | High |
| GPU | No | No | No (optional for deep models) |
| Outcome | One notebook, simple trend + ROP | Modular repo, backtest, KPI replay, dashboard | Production-style service with monitoring |
| Best for quick strong proof? | Too thin | ✅ **Yes** | Takes too long, risk of unfinished repo |

## C. Selected approach
**Option B.** It shows every skill recruiters look for (data cleaning, leakage-safe features, honest backtesting, intermittent demand, inventory maths, dashboard, tests, Git) without heavy dependencies. Everything runs on a laptop CPU in ~2 minutes.

---
## D. Architecture
```
┌────────────────┐   ┌──────────────────┐   ┌───────────┐
│ data_generator │──►│ preprocessing.py │──►│  eda.py   │──► images/02–06
│ (simulator)    │   │ QC + stock-out   │   └─────┬─────┘
└────────────────┘   │ censoring fix    │         │ p_zero (intermittency)
                     └────────┬─────────┘         │
                              ▼                   ▼
                     ┌──────────────────┐   ┌───────────────┐
                     │  features.py     │──►│  models.py    │ snaive / SBA / RF / hybrid
                     └──────────────────┘   │  backtest()   │──► metrics, residual σ
                                            └──────┬────────┘
                                                   ▼
                                            ┌───────────────┐
                                            │ inventory.py  │ SS · ROP · EOQ · Q · status · KPI replay
                                            └──────┬────────┘
                                                   ▼
                          outputs/*.csv · images/*.png · app/streamlit_app.py
```
Data flow: raw CSV → `qty_clean` (stock-out-corrected demand) → features → models → forecasts + `rmse` per series → inventory engine → recommendations → dashboard.

| Module | Responsibility |
|---|---|
| `config.py` | Paths, horizon, service level, thresholds |
| `data_generator.py` | Simulates demand + a real replenishment process |
| `preprocessing.py` | QC report, dedupe, negatives, censored-demand repair |
| `eda.py` | P0 intermittency, promo lift |
| `features.py` | Vectorised training features + single-row recursive features (same definitions) |
| `models.py` | Seasonal naive, Croston-SBA, RF, hybrid, backtest, MAE/RMSE/MASE/WAPE |
| `inventory.py` | Policy maths, recommendations, KPI replay |
| `visualize.py` | All charts |
| `main.py` | Orchestration |

---
## E. Folder structure
```
Retail-Sales-Forecasting-Inventory-Optimization/
├── README.md              # recruiter-facing landing page
├── requirements.txt       # dependencies
├── .gitignore             # ignores venv, caches, model pickles
├── main.py                # run everything
├── src/                   # importable modules (see D)
├── app/streamlit_app.py   # dashboard
├── tests/test_core.py     # unit tests for formulas & models
├── data/raw/              # simulated dataset (committed – small)
├── outputs/               # generated CSVs (forecasts, metrics, POs)
├── images/                # PNG proof for README
├── models/                # saved model (git-ignored)
├── notebooks/             # optional exploration notebook
├── reports/               # optional PDF / slides
└── docs/PROJECT_GUIDE.md  # this file
```

## F. Installation
**Python 3.10 or newer** (check `python --version`).

| Step | Windows (PowerShell/CMD) | Mac / Linux |
|---|---|---|
| Create env | `python -m venv venv` | `python3 -m venv venv` |
| Activate | `venv\Scripts\activate` | `source venv/bin/activate` |
| Install | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Verify | `python -c "import pandas, sklearn, scipy, streamlit; print('OK')"` | same |
| Freeze (optional) | `pip freeze > requirements.txt` | same |

PowerShell blocks activation? Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

---
## G. Full code, file by file
All code is in the project folder (open the files — each has comments and a docstring). Summary of what each does and how it improves on the sketch in the brief:

| File | What it does | Upgrade over the starter script |
|---|---|---|
| `src/data_generator.py` | Creates 26k rows: trend, seasonality, festivals, promos, Poisson noise, real (s,S) replenishment → genuine censored stock-outs | Starter assumed a CSV existed |
| `src/preprocessing.py` | QC report; repairs stock-out days with `max(sales, interpolation)` | Dropping stock-out rows (starter) breaks lags/rolling windows |
| `src/features.py` | Lags 1/7/14, rolling mean/std 7/14/28 (shifted), calendar, holiday, promo, encodings | Same code path for training and recursive forecasting |
| `src/models.py` | Seasonal naive, **Croston-SBA** (fixed: proper interval handling), RF, hybrid, **time-based backtest** | Starter used `GroupShuffleSplit` which leaks time; we hold out the *last* 28 days |
| `src/inventory.py` | SS/ROP/EOQ, inventory position (on-hand + on-order), status flags, 84-day policy replay | Starter function had a Python syntax error (non-default arg after default) — fixed |
| `src/visualize.py` | 11 charts | — |
| `main.py` | Runs everything, prints tables | — |
| `app/streamlit_app.py` | Select store/item, edit on-hand / on-order / lead time / service level → live ROP, EOQ, order qty | — |
| `tests/test_core.py` | Tests Croston, seasonal naive, formulas, simulation bounds | — |

**Simplified → advanced upgrades:** swap `RandomForestRegressor` for `LightGBM` in `train_rf`; add rolling-origin CV by calling `backtest` with several cutoffs; add quantile models for direct σ estimates.

---
## H. Virtual simulation workflow
**How reality is represented**
- *Products/categories:* 12 SKUs in 5 categories with price, cost, lead time, shelf life.
- *Stores:* 3 cities with size multipliers (1.0 / 1.4 / 1.2).
- *Dates:* 2023-01-01 → 2024-12-31 daily.
- *Demand variation:* Poisson noise on top of `λ`, with log-normal wobble.
- *Seasonality & trend:* day-of-week weights (weekend peak), yearly sine, +12%/yr growth, festive uplift (mid-Oct–mid-Nov, Sankranti, national holidays), summer uplift for beverages, chocolate festive spike.
- *Promotions:* ~12% of weeks at 10/15/20% discount, lift = 1 + 2.5 × discount.
- *Stock-in / stock-out:* simulator runs an (s, S) policy with lead times; when demand > stock, `stockout_flag = 1` and sales are capped (censoring).
- *Forecast → decisions → output:* hybrid forecast → SS/ROP/EOQ → order if inventory position ≤ ROP → CSV, charts, dashboard.

**Step-by-step**
1. `python main.py --regenerate` creates `data/raw/retail_sales.csv`.
2. Read the data-quality printout (duplicates, missing, stock-out rate).
3. Stock-out repair creates `qty_clean`.
4. EDA charts show trend, seasonality, promo lift, intermittency.
5. Features are engineered; RF trained on the past only.
6. Backtest on the last 28 days → accuracy table.
7. Retrain on all data; forecast Jan 2025.
8. Compute SS/ROP/EOQ per store-item → `inventory_recommendations.csv`.
9. Replay last 84 days: naive vs optimised policy → KPI chart.
10. Launch dashboard; change on-hand / lead time to demonstrate what-if decisions (e.g. set on-hand to 5 → status turns CRITICAL).

**Proof to capture:** see M.

---
## I. How to run
```bash
python main.py --regenerate     # first time (creates data + everything)
python main.py                  # re-run on existing data
streamlit run app/streamlit_app.py
python -m pytest -q
```
**Success looks like:** the terminal prints 7 numbered stages ending in `Done.`; `outputs/` has 10 CSVs, `images/` has 11 PNGs, `models/rf_model.pkl` exists.

**Sample terminal output (actual)**
```
[3/7] EDA charts saved to images/. Intermittent series (P0>0.3): 3/36
[4/7] Backtesting on the last 28 days ...
          MAE   RMSE   MASE  WAPE_%
snaive  4.190  5.128  0.761  28.232
sba     4.230  5.318  0.743  28.502
rf      3.849  4.817  0.694  25.933
hybrid  3.781  4.676  0.684  25.477
[7/7] Inventory policy replay over last 84 days:
            fill_rate  stockout_days  avg_inventory_value
naive          0.9788             77            1.00e+06
optimised      0.9920             28            1.10e+06
```
**Sample recommendation table (`outputs/inventory_recommendations.csv`)**

| store | item | on_hand | on_order | lead | μ_L | SS | ROP | EOQ | order_qty | status |
|---|---|---|---|---|---|---|---|---|---|---|
| S01 | LED Bulb 9W | 4 | 11 | 8 | 11.6 | 6.2 | 17.8 | 206 | 207 | REORDER |
| S02 | Olive Oil 500ml | 8 | 10 | 9 | 12.3 | 6.4 | 18.7 | 78 | 79 | REORDER |
| S02 | Face Wash 100ml | 15 | 0 | 7 | 11.4 | 6.0 | 17.4 | 139 | 140 | REORDER |
| S02 | Basmati Rice 5kg | 46 | 163 | 5 | 119.3 | 18.6 | 138.0 | 334 | 0 | OK |

---
## J. GitHub upload steps
1. Create account/repo: github.com → **New repository** → name `Retail-Sales-Forecasting-Inventory-Optimization`, Public, **do not** add README/.gitignore (we have them).
2. Install Git (`git --version`), then once: `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`.
3. In the project folder:
```bash
git init
git add .
git commit -m "feat: initial project structure and simulated retail dataset"
git branch -M main
git remote add origin https://github.com/<username>/Retail-Sales-Forecasting-Inventory-Optimization.git
git push -u origin main
```
4. Authentication: GitHub no longer accepts passwords → use a **Personal Access Token** (Settings → Developer settings → Tokens) or GitHub Desktop / `gh auth login`.
5. Repo settings: **Description:** *End-to-end retail demand forecasting + inventory optimization (safety stock, ROP, EOQ) with hybrid ML/Croston models and Streamlit dashboard.* **Topics:** `demand-forecasting`, `inventory-optimization`, `retail-analytics`, `supply-chain`, `time-series`, `machine-learning`, `python`, `streamlit`, `data-science`, `portfolio-project`.
6. **Pin** this repo on your profile (Profile → Customize your pins). Add a 20–30 s GIF of the dashboard (`images/dashboard_demo.gif`, record with ScreenToGif / Kap / LICEcap) and embed in README: `![demo](images/dashboard_demo.gif)`. For a longer video, upload to YouTube (unlisted) and link it.
7. Add a LICENSE (MIT) via *Add file → Create new file → LICENSE → choose template*.

**Commit message style:** `type: short imperative summary` (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).

---
## K. README
See `README.md` (complete, with results and screenshots section).

---
## L. Commit strategy / 8-day proof plan
Commit daily so the history shows genuine progression (only commit what you really did that day).

| Day | Work | Commit | Message | Save as proof | Resume/LinkedIn note |
|---|---|---|---|---|---|
| 1 | Setup, folders, venv, requirements | `.gitignore`, `requirements.txt`, `src/config.py`, README stub | `chore: project scaffold and environment` | Folder tree screenshot | "Defined scope: SKU-store daily, 28-day horizon" |
| 2 | Simulator + preprocessing | `data_generator.py`, `preprocessing.py`, dataset | `feat: simulated retail dataset and data-quality checks` | `01_dataset_preview.png`, QC printout | "Simulated 26k records with censored stock-outs" |
| 3 | EDA | `eda.py`, `visualize.py` (EDA part) | `feat: EDA – trend, seasonality, promo lift, intermittency` | images 02–06 | "Found 3/36 intermittent SKUs → needs Croston" |
| 4 | Features | `features.py` | `feat: leakage-safe lag/rolling/calendar features` | Feature table screenshot | "Shifted rolling windows to avoid leakage" |
| 5 | Forecast models + backtest | `models.py` | `feat: hybrid forecaster (RF + Croston-SBA) with backtest` | images 07–09, metrics table | "Cut WAPE 28.2→25.5% vs seasonal naive" |
| 6 | Inventory logic + tests | `inventory.py`, `tests/` | `feat: safety stock, ROP, EOQ and KPI replay` | recommendation table, images 10–12 | "Stock-out days −64% in policy replay" |
| 7 | Dashboard | `app/streamlit_app.py` | `feat: Streamlit replenishment dashboard` | `13_dashboard.png`, GIF | "Built planner what-if tool" |
| 8 | Docs & polish | README, docs, LICENSE | `docs: README, guide and results` | Repo + README screenshot | Publish LinkedIn post |

---
## M. Screenshots / proof checklist

| Asset | Generated by | File name (save in `images/`) | README reference |
|---|---|---|---|
| Dataset preview (`df.head()` in terminal/VS Code) | manual | `01_dataset_preview.png` | `![](images/01_dataset_preview.png)` |
| Missing / QC printout | `python main.py` terminal | `01b_data_quality.png` | same pattern |
| Sales trend | auto | `02_sales_trend.png` | ✓ |
| Category sales | auto | `03_category_sales.png` | ✓ |
| Seasonality | auto | `04_seasonality.png` | ✓ |
| Intermittency | auto | `05_intermittency.png` | ✓ |
| Promo lift | auto | `06_promo_lift.png` | ✓ |
| Actual vs predicted | auto | `07_actual_vs_predicted.png` | ✓ |
| Model comparison | auto | `08_model_comparison.png` | ✓ |
| Forecast graph | auto | `09_future_forecast.png` | ✓ |
| Inventory status / reorder alerts | auto | `10_inventory_status.png`, `11_top_reorders.png` | ✓ |
| KPI comparison | auto | `12_kpi_comparison.png` | ✓ |
| Recommendation table | open `inventory_recommendations.csv` in Excel | `13_recommendation_table.png` | manual |
| Dashboard | Streamlit | `14_dashboard.png` + GIF | manual |
| GitHub repo & README preview | browser | `15_repo.png`, `16_readme_preview.png` | optional (or use in LinkedIn post) |

Tables to keep in repo: everything in `outputs/`. Videos: 30-60 s screen recording of `python main.py` + dashboard what-if.

---
## N. Resume / LinkedIn / interview

### Resume bullets (numbers are from the seeded simulation — say "simulated")
1. Built an end-to-end retail demand-forecasting and replenishment pipeline (Python, scikit-learn, Streamlit) on a simulated 36-series, 26k-row dataset; hybrid Random Forest + Croston-SBA model reduced WAPE from 28.2% to 25.5% versus a seasonal-naive baseline on a time-based holdout.
2. Engineered leakage-safe lag/rolling/calendar features and corrected stock-out-censored demand, improving data integrity for 1,062 stock-out days.
3. Implemented safety-stock, reorder-point and EOQ logic at 95% service level; policy replay cut stock-out days by 64% (77→28) and lifted fill rate from 97.9% to 99.2% for ~10% more inventory value; shipped a planner dashboard with what-if ordering.

### LinkedIn descriptions
1. *Short:* "Built a Retail Sales Forecasting & Inventory Optimization system — forecast demand, calculate safety stock/reorder points/EOQ, and recommend purchase orders. Python | scikit-learn | Streamlit. 🔗 GitHub link."
2. *Business angle:* "Stock-outs lose sales; overstock locks cash. I built a simulated retail environment and a forecasting-to-replenishment engine that cut stock-out days by 64% in a policy replay. Lessons: censored demand, intermittent SKUs, and why backtests must be time-based."
3. *Learning angle:* "Project #N: from raw sales to purchase orders. Learned Croston/SBA, MASE vs WAPE, leakage-safe features, and EOQ maths. Code, charts and dashboard on GitHub — feedback welcome!"

### Explain to HR (30 seconds)
"Shops need to know how much of each product to order. I created a practice shop with fake-but-realistic sales data, predicted future demand, and calculated when and how much to reorder so that shelves rarely go empty without piling up extra stock."

### Explain to a technical interviewer
"Daily store×SKU data; stock-out days censor demand, so I repair them rather than drop them. Features are shifted lags/rolling stats plus calendar and promo. A global Random Forest is used for regular SKUs, Croston-SBA for series with P0>0.3, blended with seasonal naive. Evaluated by a time-based 28-day holdout with MAE/RMSE/MASE/WAPE. Forecast error σ feeds `SS = zσ√L`, ROP, EOQ, with orders triggered on inventory position. I validated policy by replaying 84 days."

### Business value statement
"Better forecasts let a planner hold less safety stock for the same service level, or hit higher service for the same stock — improving availability and working capital."

### 10 interview Q&A
1. **Why not drop stock-out rows?** Dropping breaks continuity of lags/rolling windows and still biases demand low; I repaired them with max(sales, interpolation). Better: model censoring explicitly (e.g. Tobit-style).
2. **Why Croston for intermittent items?** RF/ARIMA smooth toward zero or over-predict; Croston separately smooths demand size and inter-demand interval; SBA removes its upward bias by (1−α/2).
3. **How did you avoid leakage?** Rolling windows use `shift(1)`; the backtest trains only on dates before the cutoff; future promos are assumed unknown.
4. **Why MASE and WAPE, not MAPE?** MAPE explodes on zeros. MASE scales by naive error (<1 beats naive); WAPE aggregates by volume and is business-friendly.
5. **Explain safety stock.** Buffer against forecast error during lead time: `z·σ_e·√L`; z from service level (95% → 1.645).
6. **ROP vs EOQ?** ROP says *when* to order; EOQ says *how much*, balancing ordering cost K and holding cost H = unit_cost × holding rate.
7. **Why inventory position instead of on-hand?** Items already on order will arrive; ignoring them causes duplicate POs.
8. **Limitations?** Simulated data; normal-error assumption; σ√L assumes independent daily errors; promos not forecast; fixed lead time; single echelon.
9. **How would you deploy it?** Nightly batch writes recommendations to a table, API for on-demand queries, monitoring MAE/MASE and drift, monthly retrain.
10. **How do you know the policy is better?** Replayed 84 days: same EOQ and start stock, only ROP differs; safety stock cut stock-out days 77→28 and raised fill rate to 99.2% at +9.6% inventory value — a trade-off a planner can price.

---
## O. Future improvements
| Idea | How to upgrade |
|---|---|
| Multi-store / region | Add `region` feature; hierarchical forecasting + reconciliation |
| Promotion modelling | Feed *planned* promos/discount into recursive forecast; estimate uplift |
| Price elasticity | log-log regression of qty on price per category |
| Weather/events | Add weather & event-calendar features |
| Better models | LightGBM, Prophet, quantile regression for direct service-level quantiles |
| Real-time dashboard | Streamlit + scheduled refresh; FastAPI `/replenishment` |
| Automated replenishment | Auto-create PO files, supplier MOQ and case-pack constraints |
| ERP integration | Read stock/on-order from SAP/Oracle/Odoo APIs |
| Anomaly detection | Flag residual z-score > 3 or sudden P0 shifts |
| MLOps | MLflow runs, PSI drift checks, monthly retrain, GitHub Actions CI for tests |
| Real data | Kaggle: *Store Item Demand Forecasting*, *M5 Forecasting – Accuracy*, *Rossmann* — plug into the same `preprocessing` interface |

---
## P. Troubleshooting

| Problem | Reason | Solution | Prevention |
|---|---|---|---|
| `python` not found | Not on PATH | Reinstall tick "Add to PATH"; try `py` (Win) / `python3` | Verify with `python --version` |
| PowerShell won't activate venv | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` | Use CMD or Git Bash |
| `pip install` fails / version conflicts | Old pip, mixed envs | `python -m pip install -U pip`; use a fresh venv | One venv per project |
| `ModuleNotFoundError: src` | Running from wrong folder | `cd` to project root; run `python main.py` | Always run from root |
| `FileNotFoundError: retail_sales.csv` | Dataset not generated | `python main.py --regenerate` | Keep sample data committed |
| Dates load as strings | Missing `parse_dates` | `pd.read_csv(..., parse_dates=["date"])` or `pd.to_datetime` | Check `df.dtypes` |
| `KeyError` after feature step | Typo/missing column | Print `df.columns`; check `FEATURE_COLS` | Assert required columns early |
| RF training very slow | Too many trees/rows | Lower `n_estimators`, `max_depth` | Start small |
| Forecast is flat / constant | Recursive forecast not updating history, or intermittent series (Croston is flat by design) | Check `hist.append`; flat for SBA is expected | Plot early |
| Great accuracy that vanishes later | Leakage / random split on time data | Use time-based split; `shift(1)` on rolling | Backtest on last period only |
| Negative order quantity | Missing `max(0, …)` or ignoring position | Use `qty = max(EOQ, ROP − position)` only when position ≤ ROP | Unit-test formulas |
| ROP huge / SS = 0 | σ missing or lead time in wrong units | Check `resid_std`, make lead time days = forecast periods | Print intermediate values |
| `matplotlib` figure not showing | Headless backend | Images are saved to `images/`; open files | Use `plt.savefig` |
| `streamlit` page error "Run main.py first" | Outputs absent | `python main.py` | README order of steps |
| `git push` rejected (non-fast-forward) | Remote has commits (README created on GitHub) | `git pull --rebase origin main` then push | Create empty repo |
| `Authentication failed` | Password auth removed | Use PAT or `gh auth login` | Set up credentials first |
| Large/unwanted files committed | No `.gitignore` | `git rm -r --cached venv models` then commit | Add `.gitignore` before first commit |
| Images don't show in README | Wrong relative path / case | Use `images/name.png` exactly (Linux paths are case-sensitive) | Preview README on GitHub |
