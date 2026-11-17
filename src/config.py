from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Dataset Keras (Jena Climate)
JENA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)
JENA_CSV = DATA_RAW / "jena" / "jena_climate_2009_2016.csv"