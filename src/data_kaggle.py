import re

import numpy as np
import pandas as pd

from src.config import DATA_RAW

TARGET_CITY = "basel"  # thành phố dùng làm mục tiêu dự báo (dataset có 18 thành phố)
LABEL_HINTS = ("weather", "condition", "label", "class", "type", "summary", "target")
DROP_SUBSTR = ("kelvin", "fahrenheit")  # trùng thông tin với cột Celsius
DROP_EXACT = {"id", "index", "s_no", "sno", "no", "stt"}
RAIN_THRESHOLD = 0.0  # chỉ dùng khi dataset không có cột nhãn


def _clean(name):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")


def find_csv():
    files = list((DATA_RAW / "kaggle").rglob("*.csv"))
    if not files:
        raise FileNotFoundError("Không thấy file .csv nào trong data/raw/kaggle/")
    return max(files, key=lambda p: p.stat().st_size)


def build_time_features(month, hour=None):
    month = np.asarray(month)
    out = {
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }
    if hour is not None:
        hour = np.asarray(hour)
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def _parse_date(s):
    # ngày dạng số 20000101 (YYYYMMDD) phải parse theo format, nếu không sẽ bị hiểu là nano-giây
    if pd.api.types.is_numeric_dtype(s) and s.between(19000101, 21001231).all():
        return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def load_kaggle(verbose=True):
    path = find_csv()
    df = pd.read_csv(path)
    df.columns = [_clean(c) for c in df.columns]
    keep = [
        c for c in df.columns
        if not c.startswith("unnamed")
        and c not in DROP_EXACT
        and not any(s in c for s in DROP_SUBSTR)
    ]
    df = df[keep].copy()

    # --- cột ngày ---
    date_col = None
    for c in df.columns:
        if "date" in c or "time" in c:
            parsed = _parse_date(df[c])
            if parsed.notna().mean() > 0.9:
                df[c] = parsed
                date_col = c
                break
    if date_col:
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # --- cột nhãn (chữ, ít giá trị khác nhau) ---
    non_num = [
        c for c in df.columns
        if c != date_col and not pd.api.types.is_numeric_dtype(df[c])
    ]
    cands = [c for c in non_num if 2 <= df[c].nunique() <= 20]
    label_col = next(
        (c for c in cands if any(h in c for h in LABEL_HINTS)),
        cands[0] if cands else None,
    )

    num_cols = [
        c for c in df.columns
        if c != date_col
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() > 1
        and not (date_col and c == "month")  # đã có month_sin/month_cos
    ]

    focus = num_cols[:12]
    if label_col:
        mode, source = "label", label_col
        df["target"] = df[label_col].astype(str)
    else:
        rain_cands = [c for c in num_cols if "precip" in c or "rain" in c]
        if not rain_cands:
            raise ValueError(
                "Không có cột nhãn và cũng không có cột lượng mưa. "
                f"Các cột hiện có: {list(df.columns)}"
            )
        rain_col = next((c for c in rain_cands if c.startswith(TARGET_CITY)), rain_cands[0])
        mode, source = "derived", rain_col
        nxt = df[rain_col].shift(-1)
        keep_rows = nxt.notna()
        df = df[keep_rows].copy()
        df["target"] = np.where(nxt[keep_rows] > RAIN_THRESHOLD, "rain", "no_rain")
        prefix = rain_col.split("_")[0] + "_"
        focus = [c for c in num_cols if c.startswith(prefix)] or num_cols[:12]

    # --- đặc trưng thời gian ---
    time_feats = []
    if date_col:
        dt = df[date_col].dt
        hour = dt.hour.values if dt.hour.nunique() > 1 else None
        tf = build_time_features(dt.month.values, hour)
        for k, v in tf.items():
            df[k] = v
        time_feats = list(tf.keys())

    features = num_cols + time_feats
    cols = ([date_col] if date_col else []) + features + ["target"]
    out = df[cols].copy()
    if date_col:
        out = out.rename(columns={date_col: "date"})

    meta = {
        "file": path.name,
        "mode": mode,
        "source": source,
        "has_date": bool(date_col),
        "base_features": num_cols,
        "focus_features": focus,
        "time_features": time_feats,
        "features": features,
    }

    if verbose:
        print(f"[Kaggle] File: {path.name} | {len(out)} dòng")
        if date_col:
            print(f"[Kaggle] Cột ngày: {date_col} ({out['date'].min().date()} -> {out['date'].max().date()})")
        if mode == "label":
            print(f"[Kaggle] Dùng cột nhãn có sẵn: '{source}'")
        else:
            print(f"[Kaggle] Không có cột nhãn -> tạo nhãn 'ngày mai có mưa' từ '{source}'")
        print(f"[Kaggle] {len(features)} đặc trưng | nhóm trọng tâm: {focus}")
        print(out["target"].value_counts().to_string())
    return out, meta