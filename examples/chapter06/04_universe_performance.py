"""06-3. 전체 기초자산 성과: 비용 차감·연도별·월물별·시장국면별·유동성별.

선행: 01_build_universe.py
실행: uv run python examples/chapter06/04_universe_performance.py
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from hedging.backtest.report import regime_table, trade_stats
from hedging.backtest.walkforward import pool_trades
from hedging.data.universe import load_universe
from hedging.strategies.basis_rules import BasisRuleParams

pd.set_option("display.width", 200)
mapping, feats = load_universe()
# 이전상장 종목은 시장별로 두 줄이므로 최근 시장 한 줄만 남긴다(중복 집계 방지)
info = mapping.sort_values("last").drop_duplicates("code", keep="last").set_index("code")[
    ["und_name", "market", "listed_now"]]
print(f"분석 대상 {len(feats)}종목 (현물 종가 검증 통과), 기간 "
      f"{min(f['date'].min() for f in feats.values()):%Y-%m-%d} ~ {max(f['date'].max() for f in feats.values()):%Y-%m-%d}")

base = BasisRuleParams(entry_pct=0.5, exec="next_open", min_volume=1_000)
rules = {"종가 체결": replace(base, exec="close"), "다음 날 시가": base,
         "다음 날 시가·슬리피지 1틱": replace(base, slip_ticks=1.0)}
pool = pool_trades(feats, rules)

# 신호일의 국면·유동성 붙이기
reg = pd.concat([regime_table(f).assign(und_code=c) for c, f in feats.items()], ignore_index=True)
vol = pd.concat([f[["date", "front_volume"]].assign(und_code=c) for c, f in feats.items()], ignore_index=True)
pool = pool.merge(reg, left_on=["und_code", "signal_date"], right_on=["und_code", "date"], how="left")
pool = pool.merge(vol, left_on=["und_code", "signal_date"], right_on=["und_code", "date"], how="left",
                  suffixes=("", "_v"))
pool = pool.join(info, on="und_code")
pool["liq"] = pd.cut(pool["front_volume"], [0, 5_000, 50_000, np.inf], labels=["1천~5천", "5천~5만", "5만 이상"])
pool["year"] = pool["entry_date"].dt.year


def table(df, by):
    return pd.DataFrame({k: trade_stats(g) for k, g in df.groupby(by, observed=True)}).T


print("\n[1] 체결 가정별 (진입 0.5%, 신호일 최근월물 거래량 1,000계약 이상)")
print(table(pool, "rule").to_string())
print(f"  거래가 1건 이상 나온 종목: {pool[pool['rule'] == '다음 날 시가']['und_code'].nunique()}종목")

m = pool[pool["rule"] == "다음 날 시가"]
for title, by in [("[2] 연도별", "year"), ("[3] 시장별", "market"), ("[4] 신호일 거래량별", "liq"),
                  ("[5] 월물 종류별", "month_type"), ("[6] 변동성 국면별", "vol_regime"),
                  ("[7] 60일 추세 국면별", "trend_regime"), ("[8] 금리 국면별", "rate_regime"),
                  ("[9] 현재 상장 여부별(생존자 편향 점검)", "listed_now")]:
    print(f"\n{title} (다음 날 시가)")
    print(table(m, by).to_string())

top = m.groupby("und_name")["net_bp"].agg(["size", "mean"]).query("size >= 10").sort_values("mean")
print("\n[10] 거래 10건 이상 종목: 거래당 bp 하위·상위 3")
print(pd.concat([top.head(3), top.tail(3)]).round(1).to_string())
pool.to_parquet("data/processed/ch06_pool_trades.parquet", index=False)
