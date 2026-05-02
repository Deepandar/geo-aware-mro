from pathlib import Path
import pandas as pd
import duckdb

DATA_DIR = Path("data/raw/m5")
DB_PATH = "data/processed/m5.duckdb"


def load_csv(file_name: str) -> pd.DataFrame:
    path = DATA_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def profile_df(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    print("Shape:", df.shape)
    print("Nulls:\n", df.isnull().sum().head())
    print("Dtypes:\n", df.dtypes.head())


def ingest_to_duckdb():
    sales = load_csv("sales_train_validation.csv")
    calendar = load_csv("calendar.csv")
    prices = load_csv("sell_prices.csv")

    profile_df(sales, "sales")
    profile_df(calendar, "calendar")
    profile_df(prices, "prices")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB_PATH)

    con.register("sales_df", sales)
    con.register("calendar_df", calendar)
    con.register("prices_df", prices)

    con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM sales_df")
    con.execute("CREATE OR REPLACE TABLE calendar AS SELECT * FROM calendar_df")
    con.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM prices_df")

    con.close()


if __name__ == "__main__":
    ingest_to_duckdb()
