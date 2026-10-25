"""03-3. 최근월물 연속 시계열 만들기 + 품질 리포트 + DuckDB 조회.

선행: 03_collect_daily.py
실행: uv run python examples/chapter03/04_build_dataset.py [005930,000660]
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

from hedging.data.collect import load_futures, load_spot
from hedging.data.dataset import front_month_series, quality_report

unds = (sys.argv[1] if len(sys.argv) > 1 else "005930,000660").split(",")
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

for und in unds:
    fut, spot = load_futures(und), load_spot(und)
    series = front_month_series(fut, spot, roll_days=0)
    series.insert(0, "und_code", und)
    series.to_parquet(out / f"{und}_front.parquet", index=False)
    print(f"=== {und} 품질 리포트")
    for k, v in quality_report(series, fut, spot).items():
        print(f"  {k}: {v}")

print("\n=== DuckDB: 연도별 최근월물 베이시스(%) 요약")
q = """
SELECT und_code, year(date) AS year, count(*) AS days,
       round(avg(basis_pct), 3) AS avg_basis_pct, round(median(basis_pct), 3) AS med_basis_pct,
       round(avg(front_volume)) AS avg_front_vol
FROM read_parquet('data/processed/*_front.parquet')
GROUP BY ALL ORDER BY und_code, year
"""
pd.set_option("display.width", 120)
print(duckdb.sql(q).df().to_string(index=False))
