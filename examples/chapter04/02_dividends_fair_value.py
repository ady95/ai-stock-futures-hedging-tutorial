"""04-2. 배당을 반영한 이론가격과 괴리.

선행: chapter03 수집·데이터셋. 이 스크립트는 KIS 배당일정(예탁원정보)을 추가로 받는다.
실행: uv run python examples/chapter04/02_dividends_fair_value.py [005930]
"""

import sys
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from hedging.data.collect import collect_dividends
from hedging.data.kis import KisClient, yyyymmdd
from hedging.features.fair_value import dividend_table, fair_value

load_dotenv()
und = sys.argv[1] if len(sys.argv) > 1 else "005930"
series = pd.read_parquet(f"data/processed/{und}_front.parquet")
divs = collect_dividends(KisClient.from_env("real"), und, "20190101", yyyymmdd(date.today()))

pd.set_option("display.width", 140)
dt = dividend_table(divs, pd.DatetimeIndex(series["date"]))
print(f"{und} 배당 일정 (최근 8건)")
print(dt.tail(8).assign(record_date=lambda x: x.record_date.dt.strftime("%Y-%m-%d"),
                        ex_date=lambda x: x.ex_date.dt.strftime("%Y-%m-%d")).to_string(index=False))

no_div = fair_value(series, None)
exp = fair_value(series, divs, dividend_mode="expected")
act = fair_value(series, divs, dividend_mode="actual")
win = exp["div"] > 0  # 최근월물 만기 전에 배당락이 있는 날

def stat(fv, mask):
    m = fv.loc[mask, "mis_pct"]
    return f"평균 {m.mean():+.3f}%  |평균| {m.abs().mean():.3f}%  표준편차 {m.std():.3f}%"

print(f"\n배당락이 만기 전에 있는 날: {int(win.sum())}일 / {len(exp)}일")
for name, fv in [("배당 무시", no_div), ("예상 배당", exp), ("실제 배당(사후)", act)]:
    print(f"  {name:<10} 배당 구간 {stat(fv, win)}")
    print(f"  {'':<10} 그 외 구간 {stat(fv, ~win)}")

# 배당 예상이 크게 빗나간 기준일
miss = dt[(dt["status"] == "확정") & (dt["actual_dps"] != dt["expected_dps"])]
print("\n예상 배당과 실제 배당이 다른 기준일")
print(miss.assign(record_date=lambda x: x.record_date.dt.strftime("%Y-%m-%d"))
      [["record_date", "kind", "expected_dps", "actual_dps"]].to_string(index=False))
