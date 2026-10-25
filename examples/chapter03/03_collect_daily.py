"""03-3. 월물별 주식선물 일봉과 기초주식 일봉 수집 (KIS 실전 키, 조회 전용).

실행: uv run python examples/chapter03/03_collect_daily.py [005930,000660]
처음 실행은 수 분 걸린다. 만기 지난 월물은 캐시되므로 두 번째부터 빠르다.
"""

import sys
from datetime import date

from dotenv import load_dotenv

from hedging.data.codes import underlying_id
from hedging.data.collect import collect_spot, collect_underlying
from hedging.data.kis import KisClient, yyyymmdd
from hedging.data.kis_master import download_master, load_stock_futures
from hedging.instruments import listed_months

load_dotenv()
unds = (sys.argv[1] if len(sys.argv) > 1 else "005930,000660").split(",")
client = KisClient.from_env("real")
master = load_stock_futures(download_master())
today = date.today()
end_ym = listed_months(today)[-1]

for und in unds:
    row = master[master["und_code"] == und]
    if row.empty:
        print(f"{und}: 현재 주식선물이 상장되어 있지 않습니다")
        continue
    uid = underlying_id(row.iloc[0]["short_code"])
    s = collect_underlying(client, und, uid, (2020, 1), end_ym, today)
    spot = collect_spot(client, und, "20190101", yyyymmdd(today))
    got = s[s["rows"] > 0]
    print(f"{row.iloc[0]['und_name']}({und}) 기초자산 번호 {uid}")
    print(f"  월물 {len(s)}개 요청 → 데이터 있음 {len(got)}개, 일봉 합계 {got['rows'].sum():,}행"
          f" (새로 받음 {int((s['status'] == 'fetch').sum())}, 캐시 {int((s['status'] == 'cache').sum())})")
    print(f"  수집 결제월 {got.iloc[0]['expiry_ym']}({got.iloc[0]['code']}) ~ {got.iloc[-1]['expiry_ym']}({got.iloc[-1]['code']})")
    longest = got.loc[got["rows"].idxmax()]
    print(f"  상장 기간이 가장 긴 월물 {longest['code']}: {longest['first']:%Y-%m-%d}부터 {longest['rows']}행")
    print(f"  데이터 없는 월물: {', '.join(s[s['rows'] == 0]['expiry_ym']) or '없음'}")
    print(f"  현물 일봉 {len(spot):,}행 {spot['date'].min():%Y-%m-%d} ~ {spot['date'].max():%Y-%m-%d}")
