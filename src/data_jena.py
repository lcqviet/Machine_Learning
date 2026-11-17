import numpy as np
import pandas as pd

from src.config import JENA_CSV


def load_jena_hourly():
    df = pd.read_csv(JENA_CSV)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").sort_index()
    df.columns = [c.split(" (")[0] for c in df.columns]  # "T (degC)" -> "T"

    # lỗi cảm biến: tốc độ gió = -9999
    for c in ("wv", "max. wv"):
        df.loc[df[c] == -9999.0, c] = 0.0

    df = df.iloc[5::6].copy()  # 10 phút/lần -> 1 giờ/lần

    wv, mwv = df.pop("wv"), df.pop("max. wv")
    wd = np.deg2rad(df.pop("wd"))
    df["Wx"], df["Wy"] = wv * np.cos(wd), wv * np.sin(wd)
    df["maxWx"], df["maxWy"] = mwv * np.cos(wd), mwv * np.sin(wd)

    secs = np.asarray((df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s"))
    day, year = 24 * 3600, 365.2425 * 24 * 3600
    df["Day_sin"] = np.sin(secs * 2 * np.pi / day)
    df["Day_cos"] = np.cos(secs * 2 * np.pi / day)
    df["Year_sin"] = np.sin(secs * 2 * np.pi / year)
    df["Year_cos"] = np.cos(secs * 2 * np.pi / year)
    return df