import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import FIG_DIR
from src.data_jena import load_jena_hourly
from src.data_kaggle import load_kaggle

sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130)
    plt.close()
    print("  đã lưu", name)


def grid(n, ncols=3, h=3.2):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, h * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def eda_kaggle():
    print("\n=== EDA KAGGLE ===")
    df, meta = load_kaggle()
    base = meta["base_features"]
    focus = meta["focus_features"]  # các biến của thành phố mục tiêu

    # 1. Tổng quan
    print(f"Tổng số đặc trưng: {len(base)} | nhóm trọng tâm: {len(focus)}")
    print(df[focus].describe().T.to_string())
    print("Giá trị thiếu:\n", df.isna().sum().to_string())
    print("Dòng trùng:", df.duplicated().sum())
    df[base].describe().T.to_csv(FIG_DIR.parent / "kaggle_describe.csv")

    # 2. Nhãn
    order = df["target"].value_counts().index
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="target", order=order, ax=ax)
    total = len(df)
    for p in ax.patches:
        ax.annotate(f"{p.get_height() / total:.1%}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Phân bố nhãn")
    save("kaggle_label.png")

    # 3. Phân phối
    cols = focus[:12]
    fig, axes = grid(len(cols))
    for ax, c in zip(axes, cols):
        sns.histplot(df[c], kde=True, ax=ax)
        ax.set_title(c)
    save("kaggle_distribution.png")

    fig, axes = grid(len(cols))
    for ax, c in zip(axes, cols):
        sns.boxplot(data=df, x="target", y=c, order=order, ax=ax)
        ax.set_title(f"{c} theo nhãn")
    save("kaggle_boxplot_by_label.png")

    # 4. Tương quan
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[focus].corr(), annot=len(focus) <= 15, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Ma trận tương quan (thành phố mục tiêu)")
    save("kaggle_correlation.png")

    classes = sorted(df["target"].unique())
    if len(classes) == 2:
        y = (df["target"] == classes[-1]).astype(int)
        tc = df[base].corrwith(y).abs().sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        tc.plot(kind="barh", ax=ax)
        ax.set_title(f"Top 15 đặc trưng tương quan với nhãn '{classes[-1]}'")
        save("kaggle_target_correlation.png")

    # 5. Thời gian
    if "date" in df.columns:
        d = df.copy()
        d["month"] = d["date"].dt.month
        sel = [c for c in focus if any(k in c for k in ("temp_mean", "precipitation", "pressure"))][:3] or focus[:3]
        monthly = d.groupby(d["date"].dt.to_period("M"))[sel].mean()
        monthly.index = monthly.index.to_timestamp()
        fig, axes = grid(len(sel), ncols=1, h=3)
        for ax, c in zip(axes, sel):
            ax.plot(monthly.index, monthly[c])
            ax.set_title(f"Trung bình theo tháng: {c}")
        save("kaggle_time_trend.png")

        prof = d.groupby("month")[sel].mean()
        fig, axes = grid(len(sel))
        for ax, c in zip(axes, sel):
            prof[c].plot(kind="bar", ax=ax)
            ax.set_title(f"Tính mùa vụ: {c}")
        save("kaggle_seasonality.png")

        ct = pd.crosstab(d["month"], d["target"], normalize="index")
        ct.plot(kind="bar", stacked=True, figsize=(9, 4))
        plt.title("Tỉ lệ nhãn theo tháng")
        save("kaggle_label_by_month.png")
    else:
        print("Dataset không có cột ngày -> bỏ qua phân tích thời gian")


def eda_jena():
    print("\n=== EDA JENA ===")
    df = load_jena_hourly()
    base = [c for c in df.columns if not c.endswith(("_sin", "_cos"))]
    print(df[base].describe().T.to_string())
    print("Giá trị thiếu:", int(df.isna().sum().sum()), "| Số dòng:", len(df))

    fig, axes = grid(len(base), ncols=4, h=2.6)
    for ax, c in zip(axes, base):
        sns.histplot(df[c], bins=40, ax=ax)
        ax.set_title(c)
    save("jena_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df[base].corr(), annot=True, fmt=".1f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Tương quan các biến Jena")
    save("jena_correlation.png")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    df["T"].plot(ax=ax, lw=0.4)
    ax.set_title("Nhiệt độ (°C) theo giờ, 2009-2016")
    save("jena_temperature.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df.groupby(df.index.hour)["T"].mean().plot(ax=axes[0], marker="o")
    axes[0].set_title("Nhiệt độ trung bình theo giờ trong ngày")
    sns.boxplot(x=df.index.month, y=df["T"], ax=axes[1])
    axes[1].set_title("Nhiệt độ theo tháng")
    save("jena_seasonality.png")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    plot_acf(df["T"], lags=72, ax=ax)
    save("jena_acf.png")

    daily = df["T"].resample("D").mean().interpolate()
    res = seasonal_decompose(daily, model="additive", period=365)
    fig = res.plot()
    fig.set_size_inches(10, 8)
    save("jena_decompose.png")


if __name__ == "__main__":
    for fn in (eda_kaggle, eda_jena):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"!! {fn.__name__} lỗi: {e!r}")