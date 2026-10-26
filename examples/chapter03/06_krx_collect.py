"""03-2. KRX 주식선물 일별매매정보 수집 (KRX_API_KEY 필요, 서비스별 이용신청 승인 필요).

거래일 달력은 03_collect_daily.py 로 받은 KIS 현물 일봉 날짜를 쓴다.
실행: uv run python examples/chapter03/06_krx_collect.py [kospi|kosdaq] [시작일=20191202] [005930]
하루 1회 호출(약 2초, 700KB)이라 7년 치는 4개 동시 호출로 수십 분 걸린다. 중단해도 받은 날짜는 남는다.
"""

import sys
import time

from dotenv import load_dotenv

from hedging.data.collect import load_spot
from hedging.data.krx import KrxClient, collect_days

load_dotenv()
market = sys.argv[1] if len(sys.argv) > 1 else "kospi"
start = sys.argv[2] if len(sys.argv) > 2 else "20191202"
cal_src = sys.argv[3] if len(sys.argv) > 3 else "005930"

days = [d.strftime("%Y%m%d") for d in load_spot(cal_src)["date"] if d.strftime("%Y%m%d") >= start]
t0 = time.time()
n = collect_days(KrxClient.from_env(), market, days, workers=4)
print(f"{market}: 거래일 {len(days)}일 중 새로 받음 {n}일 ({time.time() - t0:.0f}초)")
