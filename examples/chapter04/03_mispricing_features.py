"""04-3. 이론 베이시스 대비 괴리와 통계 피처.

선행: 02_dividends_fair_value.py (배당 일정 수집)
실행: uv run python examples/chapter04/03_mispricing_features.py [005930]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hedging.data.collect import load_dividends
from hedging.features.basis import add_features, half_life
from hedging.features.fair_value import fair_value
from hedging.instruments import tick_size

und = sys.argv[1] if len(sys.argv) > 1 else "005930"
series = pd.read_parquet(f"data/processed/{und}_front.parquet")
fv = add_features(fair_value(series, load_dividends(und), spread=0.0))
fv["mis_ticks"] = fv["mispricing"] / fv["front_close"].map(tick_size)
Path("data/processed").mkdir(parents=True, exist_ok=True)
fv.to_parquet(f"data/processed/{und}_features.parquet", index=False)

u = fv[fv["usable"]]
print(f"{und}: 전체 {len(fv)}일 중 분석 사용 {len(u)}일 "
      f"(제외: 급변 {int(fv['big_move'].sum())}, 만기 임박 {int(fv['near_expiry'].sum())}, "
      f"거래 부진 {int(fv['thin'].sum())}, 롤오버 {int(fv['roll'].sum())} — 중복 포함)")

q = u["mis_pct"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print("\n괴리율(%) 분포:", "  ".join(f"{int(k * 100)}%={v:+.3f}" for k, v in q.items()))
print(f"평균 {u['mis_pct'].mean():+.3f}%, 표준편차 {u['mis_pct'].std():.3f}%")

print("\n|괴리|가 기준을 넘는 날의 비율")
for th in [0.1, 0.2, 0.3, 0.5, 1.0]:
    print(f"  > {th:.1f}%  {(u['mis_pct'].abs() > th).mean():6.1%}")
print("틱 단위로 보면")
for k in [1, 2, 3, 5]:
    print(f"  >= {k}틱  {(u['mis_ticks'].abs() >= k).mean():6.1%}")

phi, hl = half_life(u["mis_pct"], u["front_code"])
print(f"\n같은 월물 안 괴리의 자기상관 φ={phi:.3f}, 반감기 {hl:.2f}거래일")

by_year = u.groupby(u["date"].dt.year).agg(
    days=("mis_pct", "size"), mean=("mis_pct", "mean"), std=("mis_pct", "std"),
    abs_gt_02=("mis_pct", lambda x: (x.abs() > 0.2).mean() * 100),
    ticks_med=("mis_ticks", lambda x: np.median(np.abs(x))))
print("\n연도별 괴리")
print(by_year.round(3).rename(columns={"days": "일수", "mean": "평균%", "std": "표준편차%",
                                       "abs_gt_02": "|괴리|>0.2% 비율%", "ticks_med": "|괴리| 중앙값(틱)"}).to_string())
