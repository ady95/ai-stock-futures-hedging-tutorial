"""분봉 누적 수집 — 현물(KRX)과 최근월물 선물 1분봉 (7~9장 보조, 12장).

KIS는 분봉을 최근 약 1년만 돌려준다(2026-10 실측). 그보다 오래된 분봉이 필요하면 지금부터 매일 쌓아야 한다.
저장: data/raw/kis/minutes/<YYYYMMDD>/<기초주식코드>_spot.parquet, <기초주식코드>_fut_<선물코드>.parquet

실행:
  uv run python examples/chapter03/08_collect_minutes.py                 # 최근 거래일 하루
  uv run python examples/chapter03/08_collect_minutes.py --backfill 260  # 최근 260거래일(조회 가능한 만큼)
매일 장 마감 뒤(16시 이후) 실행한다. 이미 받은 날은 건너뛴다. 그날 받은 분봉도 잠정치일 수 있으므로
--recheck 1 로 직전 하루를 다시 받아 덮어쓸 수 있다.
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from hedging.data.kis import KisClient
from hedging.data.kis_master import download_master, load_stock_futures
from hedging.instruments import listed_months

ap = argparse.ArgumentParser()
ap.add_argument("--codes", default="005930,000660")
ap.add_argument("--backfill", type=int, default=1, help="최근 몇 거래일을 받을지")
ap.add_argument("--recheck", type=int, default=0, help="최근 며칠은 이미 있어도 다시 받기")
a = ap.parse_args()

load_dotenv()
client = KisClient.from_env("real")
root = Path("data/raw/kis/minutes")
master = load_stock_futures(download_master())
today = date.today() if datetime.now().hour >= 16 else date.today() - timedelta(days=1)

# 거래일 달력: 삼성전자 일봉 날짜를 쓰고, 그 이후는 평일로 근사
spot_daily = client.stock_daily("005930", (today - timedelta(days=int(a.backfill * 1.6) + 10)).strftime("%Y%m%d"),
                                today.strftime("%Y%m%d"))
days = [d.strftime("%Y%m%d") for d in spot_daily["date"]][-a.backfill:]


def front_code(und: str, day: str) -> str | None:
    """그날의 최근월물 단축코드(옛 체계·새 체계 모두 futures_code 가 처리)."""
    from hedging.data.codes import futures_code, underlying_id

    row = master[master["und_code"] == und]
    if row.empty:
        return None
    uid = underlying_id(row.iloc[0]["short_code"])
    y, m = listed_months(datetime.strptime(day, "%Y%m%d").date())[0]
    return futures_code(uid, y, m)


done = skipped = 0
for i, day in enumerate(days):
    out = root / day
    force = i >= len(days) - a.recheck
    for und in a.codes.split(","):
        fc = front_code(und, day)
        targets = [(f"{und}_spot.parquet", lambda: client.stock_minutes(und, day))]
        if fc:
            targets.append((f"{und}_fut_{fc}.parquet", lambda fc=fc: client.stock_futures_minutes(fc, day)))
        for name, fetch in targets:
            path = out / name
            if path.exists() and not force:
                skipped += 1
                continue
            df = fetch()
            if len(df):
                out.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path, index=False)
                done += 1
print(f"{days[0]}~{days[-1]} {len(days)}거래일: 새로 저장 {done}개, 건너뜀 {skipped}개 → {root}")
