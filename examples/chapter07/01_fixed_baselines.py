"""07-1. 무헤지·50%·100% 고정 헤지 기준선.

현물 1,000주를 들고 선물 매도 계약 수를 고정한다. 매달 최근월물 만기에 롤오버한다.
선행: chapter06/01_build_universe.py
실행: uv run python examples/chapter07/01_fixed_baselines.py
"""

from pathlib import Path

import pandas as pd

from hedging import plotting
from hedging.backtest.hedge_ledger import hedge_metrics, load_hedge_frame, run_hedge
from hedging.strategies.hedge_rules import fixed

pd.set_option("display.width", 220)
results = {}
for code in ["005930", "000660"]:
    df = load_hedge_frame(code)
    rows = []
    for h in [0.0, 0.5, 1.0]:
        res = run_hedge(df, fixed(df, h))
        results[(code, h)] = res
        rolls = res[res["roll"]]
        m = hedge_metrics(res)
        m["롤오버 비용 비중"] = round(float(rolls["cost"].sum() / res["cost"].sum()), 2) if res["cost"].sum() else 0
        rows.append({"고정 h": h, **m})
    print(f"=== {code} {df['date'].min():%Y-%m-%d}~{df['date'].max():%Y-%m-%d} ({len(df)}일, 롤오버 {int(results[(code, 1.0)]['roll'].sum())}회)")
    print(pd.DataFrame(rows).to_string(index=False))

# 완전 헤지의 손익 분해: 현물·선물·비용
r = results[("005930", 1.0)]
print("\n삼성전자 완전 헤지(h=1) 누적 손익 분해(원):",
      {k: round(float(r[k].sum())) for k in ["spot_pnl", "fut_pnl", "cost", "pnl"]})

plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

Path("outputs").mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, sharey=False)
for ax, code, name in [(axes[0], "005930", "삼성전자"), (axes[1], "000660", "SK하이닉스")]:
    for h in [0.0, 0.5, 1.0]:
        res = results[(code, h)]
        ax.plot(res["date"], res["ret"].cumsum() * 100, lw=1, label=f"h={h}")
    ax.set_title(f"{name}: 현물 1,000주 + 선물 매도 고정 헤지")
    ax.set_ylabel("일별 수익률 누적합(%)")
    ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("outputs/ch07_fixed.png")
print("그림: outputs/ch07_fixed.png")
