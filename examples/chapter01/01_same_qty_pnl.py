"""01-2. 같은 수량을 동시에 보유했을 때의 손익.

실행: uv run python examples/chapter01/01_same_qty_pnl.py
"""

import pandas as pd

from hedging.pnl import CostModel, hold_both_pnl

S0, QTY, DAYS = 100_000, 100, 20
cost = CostModel()  # 수수료 0.015%, 매도세 0.20%, 차입수수료 연 3% (예시)

rows = []
for s1 in [90_000, 95_000, 100_000, 105_000, 110_000]:
    r = hold_both_pnl(S0, s1, QTY, DAYS, cost)
    rows.append({
        "청산가": s1,
        "롱 손익": r.pnl_long,
        "숏 손익": r.pnl_short,
        "가격손익 합": r.gross,
        "비용": round(r.cost),
        "순손익": round(r.net),
    })

pd.set_option("display.width", 120)
print(f"진입가 {S0:,}원, 롱 {QTY}주 + 숏 {QTY}주, 보유 {DAYS}일")
print(pd.DataFrame(rows).to_string(index=False, formatters={c: "{:,.0f}".format for c in rows[0]}))
