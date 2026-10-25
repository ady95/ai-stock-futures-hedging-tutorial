"""03-3. 주식선물 단축코드: 새 체계와 옛 체계 (API 키 불필요).

실행: uv run python examples/chapter03/02_code_mapping.py
"""

from datetime import date

from hedging.data.codes import futures_code, underlying_id
from hedging.data.kis_master import download_master, load_stock_futures
from hedging.instruments import listed_months

fut = load_stock_futures(download_master())
sam = fut[fut["und_code"] == "005930"].iloc[0]
uid = underlying_id(sam["short_code"])
print(f"삼성전자 기초자산 번호: {uid} (종목 목록의 {sam['short_code']}에서 추출)")
for y, m in [(2024, 12), (2025, 10), (2025, 12), (2026, 1), (2026, 11), (2036, 11)]:
    print(f"  {y}-{m:02d}물 → {futures_code(uid, y, m)}")

# 현재 상장 월물 전체에 대해 규칙이 맞는지 대조
live = {f"{y}{m:02d}" for y, m in listed_months(date.today())}
cur = fut[fut["expiry_ym"].isin(live)]
ok = sum(futures_code(underlying_id(r.short_code), int(r.expiry_ym[:4]), int(r.expiry_ym[4:])) == r.short_code
         for r in cur.itertuples())
print(f"\n현재 상장 {len(cur):,}개 월물 중 규칙과 일치: {ok:,}개")
