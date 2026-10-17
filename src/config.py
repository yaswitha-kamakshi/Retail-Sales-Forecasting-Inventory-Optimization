"""Central configuration: paths and business/modelling parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "retail_sales.csv"
OUT = ROOT / "outputs"
IMG = ROOT / "images"
MODELS = ROOT / "models"

START, END = "2023-01-01", "2024-12-31"   # 2 years of daily history
SEED = 42

HORIZON = 28                  # forecast horizon (days)
SERVICE_LEVEL = 0.95          # target cycle service level -> z = 1.645
INTERMITTENT_THRESHOLD = 0.30 # series with >30% zero-demand days use Croston/SBA
RF_WEIGHT = 0.7               # hybrid blend: 70% Random Forest + 30% seasonal naive
KPI_SIM_DAYS = 84             # days used to compare inventory policies
