"""Kiểm tra chéo (cross-validation) theo thời gian cho cả hai bài toán.

- Phân loại: TimeSeriesSplit 5 fold trên dữ liệu Kaggle.
- Dự báo nhiệt độ: TimeSeriesSplit 5 fold trên đặc trưng trễ của Jena.
Kết quả được lưu vào reports/ và biểu đồ vào reports/figures/.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import FIG_DIR, RANDOM_STATE
from src.data_jena import load_jena_hourly
from src.data_kaggle import load_kaggle

REPORTS = FIG_DIR.parent
N_SPLITS = 5
STRIDE = 3
HORIZON = 24
LAGS = [1, 3, 6, 12, 24, 48]

sns.set_theme(style="whitegrid")


def cls_models():
    return {
        "Baseline majority": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced")),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.9,
            colsample_bytree=0.9, n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
            verbose=-1),
    }


def cross_validate_classification():
    print("\n=== CROSS-VALIDATION: PHÂN LOẠI (TimeSeriesSplit) ===")
    df, meta = load_kaggle(verbose=False)
    features = meta["features"]
    X = df[features]
    y = LabelEncoder().fit_transform(df["target"])
    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    rows = []
    for fold, (tr, te) in enumerate(cv.split(X), 1):
        med = X.iloc[tr].median()
        Xtr = X.iloc[tr].fillna(med)
        Xte = X.iloc[te].fillna(med)
        ytr, yte = y[tr], y[te]
        sw = compute_sample_weight("balanced", ytr)
        print(f"Fold {fold}: train {len(tr)} | test {len(te)}")

        for name, model in cls_models().items():
            t0 = time.time()
            if name == "XGBoost":
                model.fit(Xtr, ytr, sample_weight=sw)
            else:
                model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            rows.append({
                "model": name,
                "fold": fold,
                "accuracy": accuracy_score(yte, pred),
                "f1_macro": f1_score(yte, pred, labels=[0, 1],
                                     average="macro", zero_division=0),
                "train_seconds": round(time.time() - t0, 2),
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(REPORTS / "cv_classification.csv", index=False)
    summary = detail.groupby("model").agg(
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        train_seconds_mean=("train_seconds", "mean"),
    ).sort_values("f1_macro_mean", ascending=False).reset_index()
    summary.round(4).to_csv(REPORTS / "cv_classification_summary.csv", index=False)
    print("\n", summary.round(3).to_string(index=False))

    order = summary["model"].tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.boxplot(data=detail, x="f1_macro", y="model", order=order,
                hue="model", palette="Blues", legend=False, ax=ax)
    sns.stripplot(data=detail, x="f1_macro", y="model", order=order,
                  color="#0F172A", size=4, ax=ax)
    ax.set_xlabel("F1-macro")
    ax.set_ylabel("")
    ax.set_title(f"Kiểm tra chéo {N_SPLITS} fold theo thời gian - phân loại")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cv_classification.png", dpi=130)
    plt.close()
    return summary


def jena_matrix(df, stride=STRIDE):
    cols = list(df.columns)
    ti = cols.index("T")
    values = df.values.astype(np.float32)
    offsets = [k - 1 for k in LAGS]
    t_idx = np.arange(max(offsets), len(values) - HORIZON, stride)
    X = np.concatenate([values[t_idx - off] for off in offsets], axis=1)
    y = values[t_idx + HORIZON, ti]
    return X, y


def cross_validate_jena():
    print("\n=== CROSS-VALIDATION: DỰ BÁO NHIỆT ĐỘ (TimeSeriesSplit) ===")
    df = load_jena_hourly()
    X, y = jena_matrix(df)
    print(f"Mẫu: {len(X)} | đặc trưng: {X.shape[1]}")
    cv = TimeSeriesSplit(n_splits=N_SPLITS, gap=max(1, HORIZON // STRIDE))

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, max_features=0.5,
            n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, n_jobs=-1,
            random_state=RANDOM_STATE, verbose=-1),
    }

    rows = []
    for fold, (tr, te) in enumerate(cv.split(X), 1):
        print(f"Fold {fold}: train {len(tr)} | test {len(te)}")
        for name, model in models.items():
            t0 = time.time()
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            rows.append({
                "model": name,
                "fold": fold,
                "mae": mean_absolute_error(y[te], pred),
                "rmse": float(np.sqrt(np.mean((y[te] - pred) ** 2))),
                "r2": r2_score(y[te], pred),
                "train_seconds": round(time.time() - t0, 2),
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(REPORTS / "cv_jena.csv", index=False)
    summary = detail.groupby("model").agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"),
        train_seconds_mean=("train_seconds", "mean"),
    ).sort_values("mae_mean").reset_index()
    summary.round(4).to_csv(REPORTS / "cv_jena_summary.csv", index=False)
    print("\n", summary.round(3).to_string(index=False))

    order = summary["model"].tolist()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.boxplot(data=detail, x="mae", y="model", order=order,
                hue="model", palette="Oranges", legend=False, ax=ax)
    sns.stripplot(data=detail, x="mae", y="model", order=order,
                  color="#0F172A", size=4, ax=ax)
    ax.set_xlabel("MAE (°C)")
    ax.set_ylabel("")
    ax.set_title(f"Kiểm tra chéo {N_SPLITS} fold theo thời gian - dự báo nhiệt độ")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cv_jena.png", dpi=130)
    plt.close()
    return summary


def main():
    print("RANDOM_STATE =", RANDOM_STATE)
    cross_validate_classification()
    cross_validate_jena()
    print("\nĐã lưu kết quả CV vào reports/ và biểu đồ vào reports/figures/.")


if __name__ == "__main__":
    main()
