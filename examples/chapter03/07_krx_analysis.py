"""03-2. KRX 데이터로 생존자 편향·KIS 누락·교차검증 확인.

선행: 06_krx_collect.py (kospi, kosdaq), 03_collect_daily.py (KIS)
실행: uv run python examples/chapter03/07_krx_analysis.py
"""

import pandas as pd

from hedging.data.collect import load_futures, load_spot
from hedging.data.kis_master import download_master, load_stock_futures
from hedging.data.krx import load_krx, universe, yearly_counts

pd.set_option("display.width", 160)
krx = pd.concat([load_krx("kospi"), load_krx("kosdaq")], ignore_index=True)
print(f"KRX 수집: {krx['date'].min():%Y-%m-%d} ~ {krx['date'].max():%Y-%m-%d}, {krx['date'].nunique()}거래일, {len(krx):,}행")

# 1) 기초자산 수와 사라진 종목 --------------------------------------------------
print("\n[1] 연도별 개별주식 기초자산 수")
print(yearly_counts(krx).to_string())
u = universe(krx)
stock = u[~u["is_etf"]]
gone = stock[~stock["listed_now"]]
print(f"\n기간 중 등장한 개별주식 기초자산 {len(stock)}종목, 현재 상장 {int(stock['listed_now'].sum())}종목, "
      f"사라진 종목 {len(gone)}종목")
print(gone.assign(first=gone["first"].dt.strftime("%Y-%m-%d"), last=gone["last"].dt.strftime("%Y-%m-%d"))
      [["market", "und_name", "first", "last"]].tail(12).to_string(index=False))

# 2) KIS 종목 목록 누락 -----------------------------------------------------------
master = load_stock_futures(download_master())
latest = krx[krx["date"] == krx["date"].max()]
lat_stock = latest[~latest["und_name"].str.startswith("ETF")]
cov = lat_stock.assign(in_kis=lat_stock["short_code"].isin(set(master["short_code"])))
print("\n[2] 최신일 KRX 월물 중 KIS 종목 목록에 있는 비율")
print(cov.groupby("market")["in_kis"].agg(["size", "sum", "mean"]).rename(
    columns={"size": "KRX 월물", "sum": "KIS 목록에 있음", "mean": "비율"}).to_string())

# 3) KIS 선물 일봉 vs KRX ----------------------------------------------------------
print("\n[3] KIS 선물 일봉 vs KRX (삼성전자·SK하이닉스)")
for und, name in [("005930", "삼성전자"), ("000660", "SK하이닉스")]:
    kis = load_futures(und)
    k = krx[krx["und_name"] == name]
    m = kis.merge(k, left_on=["date", "code"], right_on=["date", "short_code"], suffixes=("_kis", "_krx"))
    traded = m[m["volume_krx"] > 0]
    idle = m[m["volume_krx"] == 0].copy()
    prev_settle = m.sort_values("date").groupby("code")["settle"].shift(1)
    idle["prev_settle"] = prev_settle.reindex(idle.index)
    bad = traded[traded["close_kis"] != traded["close_krx"]]
    print(f"  {name}: 공통 {len(m):,}행")
    print(f"    거래 있는 날 {len(traded):,}행: 종가 일치 {(traded['close_kis'] == traded['close_krx']).mean():.2%}"
          f" (불일치 {len(bad)}행 중 KRX 종가 공란 {int(bad['close_krx'].isna().sum())}행)")
    print(f"    거래 없는 날 {len(idle):,}행: KIS '종가' = 당일 정산가 {(idle['close_kis'] == idle['settle']).mean():.1%},"
          f" = 전일 정산가 {(idle['close_kis'] == idle['prev_settle']).mean():.1%}")

# 4) KRX 기초자산 가격(SPOT_PRC) vs KIS 현물 종가 ----------------------------------
print("\n[4] KRX가 주는 기초자산 가격 vs KIS 현물(KRX) 종가")
for und, name in [("005930", "삼성전자"), ("000660", "SK하이닉스")]:
    sp = krx[krx["und_name"] == name].groupby("date")["spot"].first()
    kis_spot = load_spot(und).set_index("date")["close"]
    j = pd.concat([sp.rename("krx_spot"), kis_spot.rename("kis_close")], axis=1, join="inner")
    j["diff"] = j["krx_spot"] != j["kis_close"]
    print(f"  {name}: 일치 {int((~j['diff']).sum())}/{len(j)}일")
    for d, r in j[j["diff"]].iterrows():
        print(f"    불일치 {d:%Y-%m-%d}: KRX 기초자산 가격 {r['krx_spot']:,.0f} / KRX 현물 종가 {r['kis_close']:,.0f}")

# 5) 정산가 vs 종가 (최근월물) -------------------------------------------------------
print("\n[5] 최근월물 정산가 vs 종가 (삼성전자)")
s = pd.read_parquet("data/processed/005930_front.parquet")[["date", "front_code"]]
f = s.merge(krx[krx["und_name"] == "삼성전자"], left_on=["date", "front_code"], right_on=["date", "short_code"])
d = (f["settle"] - f["close"]).abs()
print(f"  {len(f)}일 중 정산가 = 종가 {(d == 0).mean():.2%}, 최대 차이 {d.max():,.0f}원")
print(f"  미결제약정 제공: 최근월물 평균 {f['open_interest'].mean():,.0f}계약")

# 6) 시점별 상장 종목 목록 저장 (5~6장 백테스트용) --------------------------------------
u.to_parquet("data/processed/krx_universe.parquet", index=False)
print(f"\n[6] 기초자산 상장 기간표 저장: data/processed/krx_universe.parquet ({len(u)}행)")
