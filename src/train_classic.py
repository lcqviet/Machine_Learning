import json
import re
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import (DATA_PROCESSED, FIG_DIR, MODELS_DIR, RANDOM_STATE)
from src.data_kaggle import load_kaggle

REPORTS = FIG_DIR.parent


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    df, meta = load_kaggle()
    features = meta["features"]
    df.to_csv(DATA_PROCESSED / "kaggle_clean.csv", index=False)

    # chia theo thời gian (nếu có ngày) để tránh rò rỉ dữ liệu tương lai
    if meta["has_date"]:
        cut = int(len(df) * 0.8)
        train, test = df.iloc[:cut], df.iloc[cut:]
        print(f"\nChia theo thời gian: train {len(train)} | test {len(test)}")
    else:
        train, test = train_test_split(
            df, test_size=0.2, stratify=df["target"], random_state=RANDOM_STATE)
        print(f"\nChia ngẫu nhiên có phân tầng: train {len(train)} | test {len(test)}")

    le = LabelEncoder().fit(train["target"])
    test = test[test["target"].isin(le.classes_)]
    med = train[features].median()
    Xtr, Xte = train[features].fillna(med), test[features].fillna(med)
    ytr, yte = le.transform(train["target"]), le.transform(test["target"])
    sw = compute_sample_weight("balanced", ytr)
    labels = np.arange(len(le.classes_))
    print("Các lớp:", list(le.classes_))

    models = {
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

    rows, importances, files = [], {}, {}
    for name, model in models.items():
        t0 = time.time()
        if name == "XGBoost":
            model.fit(Xtr, ytr, sample_weight=sw)
        else:
            model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(yte, pred),
            "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
            "train_seconds": round(time.time() - t0, 2),
        })
        print(f"{name:22s} acc={rows[-1]['accuracy']:.3f} "
              f"f1_macro={rows[-1]['f1_macro']:.3f}")

        fname = f"cls_{slug(name)}.joblib"
        joblib.dump(model, MODELS_DIR / fname)
        files[name] = fname

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_predictions(
            yte, pred, labels=labels, display_labels=le.classes_,
            normalize="true", cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"cm_{slug(name)}.png", dpi=120)
        plt.close()

        if hasattr(model, "feature_importances_"):
            importances[name] = model.feature_importances_

    res = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    res.to_csv(REPORTS / "results_classification.csv", index=False)
    print("\n", res.to_string(index=False))

    pd.DataFrame(importances, index=features).rename_axis("feature") \
        .reset_index().to_csv(REPORTS / "feature_importance.csv", index=False)

    best = res[res["model"] != "Baseline majority"].iloc[0]["model"]
    best_pred = models[best].predict(Xte)
    print(f"\nMô hình tốt nhất (macro-F1): {best}")
    print(classification_report(yte, best_pred, labels=labels,
                                target_names=le.classes_, zero_division=0))

    stats = {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                 "median": float(df[c].median())}
             for c in meta["base_features"]}
    meta.update({
        "classes": list(le.classes_),
        "best_model": best,
        "model_files": files,
        "feature_stats": stats,
        "medians": {k: float(v) for k, v in med.items()},
    })
    (MODELS_DIR / "cls_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu mô hình vào thư mục models/")


if __name__ == "__main__":
    main()