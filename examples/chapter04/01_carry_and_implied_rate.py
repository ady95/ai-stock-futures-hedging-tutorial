"""04-1. 실제 베이시스와 보유비용 — 시장이 쓰는 금리를 역산한다.

선행: chapter03/03_collect_daily.py, 04_build_dataset.py
실행: uv run python examples/chapter04/01_carry_and_implied_rate.py [005930]
"""

import sys

import pandas as pd

from hedging.features.fair_value import fair_value
from hedging.features.rates import base_rate

und = sys.argv[1] if len(sys.argv) > 1 else "005930"
series = pd.read_parquet(f"data/processed/{und}_front.parquet")
fv = fair_value(series, divs=None, spread=0.0)  # 배당을 빼지 않은 순수 보유비용

fv["year"] = fv["date"].dt.year
fv["base_rate"] = base_rate(fv["date"]).values * 100
g = fv.groupby("year").agg(
    days=("date", "size"),
    basis_med=("basis_pct", "median"),
    carry_med=("theo_basis_pct", "median"),
    implied_med=("implied_rate", "median"),
    base_avg=("base_rate", "mean"),
)
g["implied_med"] *= 100
print(f"{und} 최근월물: 실제 베이시스 vs 보유비용(배당 제외), 연도별 중앙값")
print(g.round(3).rename(columns={
    "days": "일수", "basis_med": "실제베이시스%", "carry_med": "보유비용%",
    "implied_med": "역산금리%", "base_avg": "기준금리%"}).to_string())
