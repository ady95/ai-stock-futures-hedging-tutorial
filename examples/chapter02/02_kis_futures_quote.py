"""02-1. KIS Open API로 기초주식과 주식선물 시세를 함께 조회한다 (조회 전용).

준비: .env 에 KIS_REAL_APP_KEY / KIS_REAL_APP_SECRET (또는 PAPER) 설정
실행: uv run python examples/chapter02/02_kis_futures_quote.py [기초자산코드=005930] [--paper]
"""

import sys

from dotenv import load_dotenv

from hedging.data.kis import KisClient
from hedging.data.kis_master import download_master, load_stock_futures
from hedging.instruments import price_limits

load_dotenv()
args = [a for a in sys.argv[1:] if not a.startswith("--")]
und = args[0] if args else "005930"
client = KisClient.from_env("paper" if "--paper" in sys.argv else "real")

fut = load_stock_futures(download_master())
contracts = fut[fut["und_code"] == und].sort_values("expiry_ym")
if contracts.empty:
    sys.exit(f"{und}: 상장된 주식선물이 없습니다")

spot = client.stock_price(und)
s = int(spot["stck_prpr"])
print(f"기초주식 {contracts.iloc[0]['und_name']}({und}) 현재가 {s:,}원")
print(f"{'종목코드':<8}{'결제월':>8}{'선물가':>11}{'F-S':>8}{'거래량':>11}{'미결제약정':>12}{'최종거래일':>11}{'잔존':>5}  가격제한(API / 계산)")
for _, c in contracts.iterrows():
    q = client.stock_futures_price(c["short_code"])
    f = int(q["futs_prpr"])
    lo, hi = price_limits(int(q["futs_sdpr"]), 0.10)
    same = "일치" if (lo, hi) == (int(q["futs_llam"]), int(q["futs_mxpr"])) else "불일치"
    print(f"{c['short_code']:<8}{c['expiry_ym']:>10}{f:>12,}{f - s:>9,}{int(q['acml_vol']):>13,}"
          f"{int(q['hts_otst_stpl_qty']):>13,}{q['futs_last_tr_date']:>13}{q['hts_rmnn_dynu']:>5}"
          f"  {int(q['futs_llam']):,}~{int(q['futs_mxpr']):,} / {same}")
