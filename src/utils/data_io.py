from pathlib import Path
import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_parquet(path)


def main() -> None:
    df = pd.DataFrame(
        [{"item_id": 1, "source": "m5", "version": "v1"}]
    )
    save_parquet(df, Path("data/interim/sample.parquet"))


if __name__ == "__main__":
    main()
