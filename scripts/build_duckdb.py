import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
DB_PATH = BASE_DIR / "data" / "muni_market.duckdb"

TRADE_FILE = DATA_DIR / "Trade_Output_Sample.csv"

con = duckdb.connect(str(DB_PATH))

con.execute(f"""
CREATE OR REPLACE TABLE trades AS
SELECT *
FROM read_csv_auto('{TRADE_FILE}', header=True)
""")

con.close()

print("DuckDB database created successfully.")
print(f"Source file: {TRADE_FILE}")
print(f"Database: {DB_PATH}")
