"""01-3. 청산 시점을 분리하면 무슨 일이 생기는가 — 대표 경로 5종.

실행: uv run python examples/chapter01/02_split_exit_scenarios.py
"""

import pandas as pd

from hedging.pnl import ZERO_COST, hindsight_pnl, split_exit_pnl
from hedging.sim import scenario_paths

QTY, TAKE_PROFIT = 100, 0.05
LEG = {"long": "롱", "short": "숏", None: "-"}

rows = []
for name, path in scenario_paths(s0=100_000, days=20).items():
    r = split_exit_pnl(path, QTY, TAKE_PROFIT, ZERO_COST)
    h = hindsight_pnl(path, QTY, ZERO_COST)
    rows.append({
        "경로": name,
        "먼저 청산": LEG[r.first_leg],
        "롱 청산일": r.exit_day_long,
        "숏 청산일": r.exit_day_short,
        "한쪽 노출일": r.one_leg_days,
        "분리청산 손익": round(r.gross),
        "사후최적 손익": round(h.gross),
    })

print(f"롱·숏 각 {QTY}주, 목표수익 {TAKE_PROFIT:.0%}, 비용 제외, 20거래일")
print(pd.DataFrame(rows).to_string(index=False, formatters={
    "분리청산 손익": "{:,}".format, "사후최적 손익": "{:,}".format}))
