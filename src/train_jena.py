import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import DATA_PROCESSED, FIG_DIR, MODELS_DIR, RANDOM_STATE
from src.data_jena import load_jena_hourly

REPORTS = FIG_DIR.parent
PAST = 120          # nhìn lại 120 giờ
HORIZON = 24        # dự báo trước 24 giờ
STRIDE_TRAIN = 3    # lấy mẫu thưa cho train/val để chạy nhanh
LAGS = [1, 3, 6, 12, 24, 48]  # các mốc (giờ trước) dùng cho mô hình bảng


def make_windows(arr, t_idx, stride):
    n = len(arr) - PAST - HORIZON + 1
    w = sliding_window_view(arr, PAST, axis=0)[:n].transpose(0, 2, 1)
    y = arr[PAST - 1 + HORIZON: PAST - 1 + HORIZON + n, t_idx]
    X = np.ascontiguousarray(w[::stride], dtype=np.float32)
    return X, y[::stride].astype(np.float32)


def tab(X):
    return X[:, [-k for k in LAGS], :].reshape(len(X), -1)


def metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p))}


def main():
    from tensorflow import keras

    keras.utils.set_random_seed(RANDOM_STATE)
    df = load_jena_hourly()
    cols = list(df.columns)
    ti = cols.index("T")
    n = len(df)
    tr, va = int(n * 0.70), int(n * 0.85)
    print(f"Số giờ: {n} | train {tr} | val {va - tr} | test {n - va}")

    values = df.values.astype("float32")
    mean, std = values[:tr].mean(0), values[:tr].std(0) + 1e-8
    norm = (values - mean) / std

    Xtr, ytr = make_windows(norm[:tr], ti, STRIDE_TRAIN)
    Xva, yva = make_windows(norm[tr:va], ti, STRIDE_TRAIN)
    Xte, yte = make_windows(norm[va:], ti, 1)
    print("Cửa sổ:", Xtr.shape, Xva.shape, Xte.shape)

    inv = lambda a: a * std[ti] + mean[ti]  # noqa: E731
    y_true = inv(yte)

    t0 = va + PAST - 1 + HORIZON
    times = df.index[t0: t0 + len(yte)]

    preds, rows = {}, []

    def record(name, p_norm, secs=0.0):
        p = inv(np.asarray(p_norm).ravel())
        preds[name] = p
        rows.append({"model": name, **metrics(y_true, p),
                     "train_seconds": round(secs, 1)})
        print(f"{name:22s} MAE={rows[-1]['mae']:.3f} °C  "
              f"RMSE={rows[-1]['rmse']:.3f}  R2={rows[-1]['r2']:.3f}")

    # Baseline: dự báo = nhiệt độ hiện tại
    record("Persistence", Xte[:, -1, ti])

    # Các mô hình trên đặc trưng trễ (lag)
    Ttr, Tva, Tte = tab(Xtr), tab(Xva), tab(Xte)
    tab_models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, max_features=0.5,
            n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, n_jobs=-1,
            random_state=RANDOM_STATE, verbose=-1),
    }
    for name, m in tab_models.items():
        t1 = time.time()
        m.fit(Ttr, ytr)
        record(name, m.predict(Tte), time.time() - t1)
        joblib.dump(m, MODELS_DIR / f"jena_{name.lower().replace(' ', '_')}.joblib")

    # LSTM
    model = keras.Sequential([
        keras.Input(shape=(PAST, len(cols))),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    t1 = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=25,
                     batch_size=128, callbacks=cb, verbose=2)
    record("LSTM", model.predict(Xte, batch_size=512, verbose=0), time.time() - t1)
    model.save(MODELS_DIR / "jena_lstm.keras")

    # Kết quả
    res = pd.DataFrame(rows).sort_values("mae")
    res.to_csv(REPORTS / "results_jena.csv", index=False)
    print("\n", res.to_string(index=False))

    out = pd.DataFrame({"time": times, "actual": y_true, **preds})
    out.to_csv(DATA_PROCESSED / "jena_test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    k = 24 * 10
    ax.plot(times[:k], y_true[:k], label="Thực tế", color="black", lw=1.5)
    for name in ("LSTM", "LightGBM", "Persistence"):
        ax.plot(times[:k], preds[name][:k], label=name, lw=1, alpha=0.85)
    ax.set_ylabel("°C")
    ax.set_title(f"Dự báo nhiệt độ trước {HORIZON} giờ (10 ngày đầu tập test)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "jena_forecast_compare.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist.history["loss"], label="train")
    ax.plot(hist.history["val_loss"], label="val")
    ax.set_title("LSTM: loss theo epoch")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "jena_lstm_loss.png", dpi=130)
    plt.close()

    meta = {"columns": cols, "mean": mean.tolist(), "std": std.tolist(),
            "past": PAST, "horizon": HORIZON, "lags": LAGS, "target": "T"}
    (MODELS_DIR / "jena_meta.json").write_text(json.dumps(meta, indent=2))
    print("Xong. Mô hình và kết quả đã lưu.")


if __name__ == "__main__":
    main()