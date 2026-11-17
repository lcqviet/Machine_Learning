import pandas as pd

from src.config import DATA_RAW


def main():
    files = sorted(DATA_RAW.rglob("*.csv"))
    if not files:
        print("Không tìm thấy file .csv nào trong data/raw")
        return

    for f in files:
        print("=" * 80)
        print("FILE:", f.relative_to(DATA_RAW))
        df = pd.read_csv(f)
        print("Kích thước:", df.shape)
        print("\nCác cột và kiểu dữ liệu:")
        print(df.dtypes)
        print("\n5 dòng đầu:")
        print(df.head().to_string())
        na = df.isna().sum()
        print("\nGiá trị thiếu:")
        print(na[na > 0].to_string() if (na > 0).any() else "Không có")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            n = df[col].nunique()
            print(f"\nCột chữ '{col}': {n} giá trị khác nhau")
            if n <= 15:
                print(df[col].value_counts().to_string())


if __name__ == "__main__":
    main()