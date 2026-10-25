"""02-2. 헤지했는데도 현금이 필요하다 — 일일정산과 추가증거금.

현물 200주 매수 + 주식선물 20계약 매도(=200주)를 들고 있는데 주가가 사흘 연속 오르는 상황.
가격손익은 상쇄되지만, 선물 손실은 매일 현금으로 정산되고 현물 이익은 팔기 전까지 현금이 아니다.
실행: uv run python examples/chapter02/03_margin_settlement.py
"""

import pandas as pd

from hedging.risk import MarginRates, daily_settlement, initial_margin

rates = MarginRates(initial=0.36, maintenance=0.24)  # 2026년 5월 삼성전자 선물 예시
N, SPOT_QTY = 20, 200
S0, F0 = 213_500, 215_000
moves = [0.0, 0.10, 0.10, 0.05, -0.03]  # 일별 등락률 (첫 값은 진입일)

f_path, s_path = [F0], [S0]
for r in moves[1:]:
    f_path.append(round(f_path[-1] * (1 + r), -2))
    s_path.append(round(s_path[-1] * (1 + r), -2))

df = daily_settlement(f_path, N, rates, side="short")
df["spot_unrealized"] = [(s - S0) * SPOT_QTY for s in s_path]
df["fut_cum_pnl"] = df["variation"].cumsum()
df["hedged_total"] = df["spot_unrealized"] + df["fut_cum_pnl"]
df["call"] = df["call"].map({True: "발생", False: "-"})

print(f"진입 위탁증거금 {initial_margin(F0, N, rates):,.0f}원 (선물 {F0:,}원 × {N}계약 × 10주 × {rates.initial:.0%})")
view = df[["day", "price", "variation", "equity", "maint_req", "call", "top_up",
           "spot_unrealized", "hedged_total"]].rename(columns={
    "day": "일", "price": "선물정산가", "variation": "정산차금", "equity": "예탁금",
    "maint_req": "유지증거금", "call": "마진콜", "top_up": "추가입금",
    "spot_unrealized": "현물평가손익", "hedged_total": "헤지합계손익"})
money = "{:,.0f}".format
print(view.to_string(index=False, formatters={c: money for c in view.columns if c not in ("일", "마진콜")}))
print(f"\n누적 추가입금 {df['cum_top_up'].iloc[-1]:,.0f}원 — 헤지합계손익은 {df['hedged_total'].iloc[-1]:,.0f}원인데도")
