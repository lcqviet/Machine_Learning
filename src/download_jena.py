import io
import urllib.request
import zipfile

from src.config import JENA_CSV, JENA_URL


def main():
    if JENA_CSV.exists():
        print(f"Đã có sẵn: {JENA_CSV}")
        return

    JENA_CSV.parent.mkdir(parents=True, exist_ok=True)
    print("Đang tải Jena Climate (~13 MB)...")
    with urllib.request.urlopen(JENA_URL) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(JENA_CSV.parent)

    print(f"Xong: {JENA_CSV}")


if __name__ == "__main__":
    main()