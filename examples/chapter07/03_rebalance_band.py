"""07-3. 거래비용을 고려한 재조정 임계값(밴드)과 계약 정수화.

목표 계약 수와 현재 계약 수의 차이가 band 계약 이상일 때만 거래한다.
실행: uv run python examples/chapter07/03_rebalance_band.py
"""

from dataclasses import replace

import pandas as pd

from hedging.backtest.hedge_ledger import HedgeConfig, hedge_metrics, load_hedge_frame, run_hedge
from hedging.strategies.hedge_rules import POLICIES

pd.set_option("display.width", 220)
cols = ["평균 h", "연수익%", "연변동성%", "CVaR5%(일)", "헤지효과(분산감소)", "연 거래 계약", "연 비용bp"]
for code in ["005930", "000660"]:
    df = load_hedge_frame(code)
    for name in ["변동성 목표", "추세+변동성"]:
        h = POLICIES[name](df)
        rows = []
        for band in [0, 2, 5, 10, 20]:
            m = hedge_metrics(run_hedge(df, h, replace(HedgeConfig(), band=band)))
            rows.append({"밴드(계약)": band, **{k: m[k] for k in cols}})
        print(f"=== {code} · {name}")
        print(pd.DataFrame(rows).to_string(index=False))

# 정수화 오차: 1,000주 기준이면 1계약 = h 1%p
df = load_hedge_frame("005930")
h = POLICIES["변동성 목표"](df)
res = run_hedge(df, h)
err = (res["h_actual"] - res["h_target"].clip(0, 1)).abs()
print(f"\n정수화 오차(현물 1,000주, 밴드 0): 평균 {err.mean() * 100:.2f}%p, 최대 {err.max() * 100:.2f}%p")
for q in [1_000, 205, 55]:
    r = run_hedge(df, h, replace(HedgeConfig(), qty=q))
    e = (r["h_actual"] - r["h_target"].clip(0, 1)).abs()
    print(f"  현물 {q:>5,}주: 평균 오차 {e.mean() * 100:5.2f}%p, 최대 {e.max() * 100:5.2f}%p")
