"""02-4. 실습: 계약 단위와 증거금 위험을 반영한 헤지 계산기.

실행 예:
  uv run python examples/chapter02/04_hedge_calculator.py --qty 203 --spot 213500 --futures 215000
  uv run python examples/chapter02/04_hedge_calculator.py --qty 203 --spot 213500 --futures 215000 --cash 25000000
"""

import argparse

from hedging.risk import (
    MarginRates,
    exposure_krw,
    initial_margin,
    maintenance_margin,
    margin_call_price,
    plan_hedge,
)

ap = argparse.ArgumentParser(description="현물–주식선물 헤지 계산기")
ap.add_argument("--qty", type=int, required=True, help="보유 현물 수량(주)")
ap.add_argument("--spot", type=float, required=True, help="현물 가격")
ap.add_argument("--futures", type=float, required=True, help="선물 가격")
ap.add_argument("--ratio", type=float, default=1.0, help="목표 헤지 비율 h (기본 1.0)")
ap.add_argument("--multiplier", type=float, default=10, help="거래승수 (기본 10, 기업행사 조정 시 변경)")
ap.add_argument("--initial", type=float, default=0.36, help="위탁증거금률 (기본 0.36)")
ap.add_argument("--maint", type=float, default=0.24, help="유지증거금률 (기본 0.24)")
ap.add_argument("--cash", type=float, default=None, help="선물 계좌 예탁금. 생략하면 위탁증거금만큼")
a = ap.parse_args()

rates = MarginRates(a.initial, a.maint)
print(f"입력: 현물 {a.qty:,}주 @ {a.spot:,.0f}원, 선물 @ {a.futures:,.0f}원, 목표 h={a.ratio}, 승수 {a.multiplier:g}")

print("\n[1] 계약 수 (반올림 방식별, 주식 수 기준)")
for mode in ["nearest", "down", "up"]:
    p = plan_hedge(a.qty, a.ratio, a.multiplier, mode)
    print(f"  {mode:<8} {p.contracts:>4}계약 = {p.hedged_shares:>6,.0f}주  순노출 {p.net_shares:>+6,.0f}주  실제 h={p.achieved_ratio:.3f}")
pv = plan_hedge(a.qty, a.ratio, a.multiplier, "nearest", a.spot, a.futures)
print(f"  금액 기준  {pv.contracts:>4}계약 (현물 평가액 ÷ 선물 1계약 명목금액)")

plan = plan_hedge(a.qty, a.ratio, a.multiplier, "nearest")
ex = exposure_krw(plan, a.spot, a.futures)
print(f"\n[2] 선택: {plan.contracts}계약 매도")
print(f"  현물 평가액     {ex['spot_value']:>15,.0f}원")
print(f"  선물 명목금액   {ex['futures_notional']:>15,.0f}원 (매도)")
print(f"  순노출 금액     {ex['net_value']:>+15,.0f}원")

if plan.contracts == 0:
    raise SystemExit("\n계약 수가 0입니다 — 보유 수량이 1계약(승수) 단위보다 작습니다.")

im = initial_margin(a.futures, plan.contracts, rates, a.multiplier)
mm = maintenance_margin(a.futures, plan.contracts, rates, a.multiplier)
cash = im if a.cash is None else a.cash
f_call = margin_call_price(a.futures, plan.contracts, rates, cash, "short", a.multiplier)
print("\n[3] 증거금")
print(f"  위탁증거금 {im:>15,.0f}원 ({a.initial:.1%})")
print(f"  유지증거금 {mm:>15,.0f}원 ({a.maint:.1%})")
print(f"  예탁금     {cash:>15,.0f}원")
if cash < im:
    print("  경고: 예탁금이 위탁증거금보다 적어 신규 주문을 낼 수 없습니다")
print(f"  마진콜 발생 선물가격 {f_call:>12,.0f}원 (현재 대비 {f_call / a.futures - 1:+.2%})")

print("\n[4] 가격 시나리오 (현물·선물이 같은 비율로 움직인다고 가정, 하루 만에)")
print(f"  {'등락':>6}{'현물손익':>14}{'선물정산차금':>14}{'합계':>12}{'예탁금':>15}  상태")
for r in [-0.10, -0.05, 0.05, 0.10, 0.20, 0.30]:
    s_pnl = a.qty * a.spot * r
    f_pnl = -plan.contracts * a.multiplier * a.futures * r
    eq = cash + f_pnl
    need = maintenance_margin(a.futures * (1 + r), plan.contracts, rates, a.multiplier)
    state = "마진콜" if eq < need else "정상"
    print(f"  {r:>+6.0%}{s_pnl:>15,.0f}{f_pnl:>15,.0f}{s_pnl + f_pnl:>13,.0f}{eq:>16,.0f}  {state}")
