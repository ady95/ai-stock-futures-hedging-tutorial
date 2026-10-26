"""06. 전체 개별주식선물 기초자산으로 데이터셋 넓히기 (KRX 상장 기간표 + KIS 현물·배당).

선행: chapter03/06_krx_collect.py (kospi, kosdaq)
실행: uv run python examples/chapter06/01_build_universe.py
처음 실행은 종목 수 × 약 20회 호출이라 수십 분 걸린다. 받은 종목은 다시 받지 않는다.
"""

import time

import pandas as pd
from dotenv import load_dotenv

from hedging.data.kis import KisClient
from hedging.data.kis_master import download_master, load_stock_futures
from hedging.data.krx import load_krx, universe
from hedging.data.universe import build_features, collect_universe, map_underlyings, stock_master

load_dotenv()
t0 = time.time()
krx = pd.concat([load_krx("kospi"), load_krx("kosdaq")], ignore_index=True)
u = universe(krx)
stocks = u[~u["is_etf"]]
masters = pd.concat([stock_master("kospi"), stock_master("kosdaq")], ignore_index=True)
fut_master = load_stock_futures(download_master())
mapping = map_underlyings(stocks["und_name"].tolist(), masters, fut_master, krx).merge(stocks, on="und_name")
miss = mapping[mapping["code"].isna()]
print(f"기초자산 {len(mapping)}종목 → 매핑 방법별: {mapping['how'].value_counts().to_dict()}")
print("  매핑 실패:", [(r.und_name, "상장중" if r.listed_now else "이탈") for r in miss.itertuples()])

start = krx["date"].min().strftime("%Y%m%d")
end = krx["date"].max().strftime("%Y%m%d")
log = collect_universe(lambda: KisClient.from_env("real"), mapping, start, end, workers=6)
print(f"현물·배당 수집: 성공 {int(log['ok'].sum())}, 실패 {int((~log['ok']).sum())}")
fail = log[~log["ok"]].merge(mapping[["code", "und_name", "how", "listed_now"]], on="code")
print("  수집 실패:", fail[["und_name", "code", "how", "listed_now"]].values.tolist())

checks = []
for code, names in mapping.dropna(subset=["code"]).groupby("code")["und_name"].unique().items():
    fv = build_features(krx, list(names), code)
    both = fv.dropna(subset=["spot", "krx_spot"]) if fv is not None else None
    for n in names:
        checks.append({"und_name": n, "code": code, "days": 0 if fv is None else len(fv),
                       "spot_match": float((both["spot"] == both["krx_spot"]).mean()) if both is not None and len(both) else None})
chk = pd.DataFrame(checks)
mapping = mapping.merge(chk, on=["und_name", "code"], how="left")
mapping["valid"] = mapping["spot_match"] >= 0.95
mapping.to_parquet("data/processed/universe_mapping.parquet", index=False)
bad = mapping[mapping["code"].notna() & ~mapping["valid"]]
ok = mapping[mapping["days"] > 0]
print(f"피처 생성: 선물 기초자산명 {len(ok)}개 / 현물 코드 {ok['code'].nunique()}종목, "
      f"현물 종가 = KRX 기초자산 가격 95% 이상(검증 통과) 현물 코드 {mapping.loc[mapping['valid'], 'code'].nunique()}종목")
moved = ok.groupby("code").agg(names=("und_name", "unique"), markets=("market", "unique"))
moved = moved[moved["markets"].map(len) > 1]
print("  두 시장에 걸친 종목(코스닥→유가 이전상장 등, 시계열을 합침):",
      [(c, list(r.names), list(r.markets)) for c, r in moved.iterrows()])
print("  검증 실패:", bad[["und_name", "code", "how", "days", "spot_match"]].round(3).values.tolist())
print(f"({time.time() - t0:.0f}초)")
