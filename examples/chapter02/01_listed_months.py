"""02-1. 상장 결제월·최종거래일 계산과 KIS 종목 마스터 대조.

실행: uv run python examples/chapter02/01_listed_months.py [YYYY-MM-DD] [--master]
  --master : KIS 주식선물 종목 마스터(공개 파일, API 키 불필요)를 받아 계산 결과와 대조
"""

import sys
from datetime import date

from hedging.instruments import last_trading_day, listed_months, price_limits, tick_size

args = [a for a in sys.argv[1:] if not a.startswith("--")]
today = date.fromisoformat(args[0]) if args else date.today()

print(f"기준일 {today} 의 상장 결제월 (비분기월 2 + 분기월 4)")
for y, m in listed_months(today):
    ltd = last_trading_day(y, m)
    kind = "분기월" if m in (3, 6, 9, 12) else "비분기월"
    print(f"  {y}{m:02d}  {kind:<4}  최종거래일 {ltd} ({'월화수목금토일'[ltd.weekday()]})  남은 날 {(ltd - today).days:>3}일")

print("\n호가가격단위와 가격제한폭(1단계 ±10%) 예시")
for base in [48_000, 213_500, 214_500, 1_226_000]:
    lo, hi = price_limits(base, 0.10)
    print(f"  기준가 {base:>9,}원  호가단위 {tick_size(base):>5,}원  하한 {lo:>9,}원  상한 {hi:>9,}원")

if "--master" in sys.argv:
    from hedging.data.kis_master import download_master, load_stock_futures

    fut = load_stock_futures(download_master())
    print(f"\nKIS 마스터: 주식선물 {len(fut):,}행, 기초자산 {fut['und_code'].nunique()}종목")
    print("  거래승수 분포:", fut["multiplier"].value_counts().to_dict())
    expected = {f"{y}{m:02d}" for y, m in listed_months(today)}
    by_month = fut["expiry_ym"].value_counts().sort_index()
    print("  결제월별 행 수:")
    for ym, n in by_month.items():
        mark = "계산과 일치" if ym in expected else "계산 목록에 없음(만기 경과 등 점검 필요)"
        print(f"    {ym}: {n:>4}  {mark}")
    stale = fut[~fut["expiry_ym"].isin(expected)]
    if len(stale):
        print(f"  주의: 계산 목록에 없는 결제월을 가진 기초자산 {stale['und_code'].nunique()}종목 — 사용 전 걸러낸다")
