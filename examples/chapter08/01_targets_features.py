"""08-1. 예측 대상 정의: 단기 방향·급락 위험·변동성.

유동성 상위 20종목(7-4)의 일별 패널을 만들고 레이블의 성격을 본다.
실행: uv run python examples/chapter08/01_targets_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hedging.backtest.hedge_ledger import load_hedge_frame
from hedging.data.universe import load_universe
from hedging.features.hedge_features import FEATURES, H, build_panel
from hedging.models.lgbm_hedge import top_liquid_codes

pd.set_option("display.width", 200)
mapping, feats = load_universe()
codes = top_liquid_codes(feats, 20)
panel = build_panel({c: load_hedge_frame(c) for c in codes})
Path("data/processed").mkdir(parents=True, exist_ok=True)
panel.to_parquet("data/processed/ch08_panel.parquet", index=False)
lab = panel.dropna(subset=["y_crash"])
print(f"패널: {len(codes)}종목 × {panel['date'].nunique()}일 = {len(panel):,}행, 피처 {len(FEATURES)}개, 예측 기간 {H}거래일")

print("\n[연도별 레이블]")
g = lab.groupby(lab["date"].dt.year)
print(pd.DataFrame({
    "행": g.size(),
    "급락 비율(y_crash)": g["y_crash"].mean().round(3),
    "상승 비율(y_up)": g["y_up"].mean().round(3),
    "미래 5일 변동성 중앙값%": (g["fwd_vol"].median() * 100).round(1),
}).to_string())

print("\n[변동성의 지속성: 오늘 20일 변동성 vs 미래 5일 변동성 (로그 상관)]")
ok = lab.dropna(subset=["vol_20", "fwd_vol"])
print(f"  상관계수 {np.corrcoef(np.log(ok['vol_20']), np.log(ok['fwd_vol']))[0, 1]:.3f}")
print(f"  미래 수익률의 자기상관(오늘 5일 수익률 vs 미래 5일 수익률) {np.corrcoef(ok['ret_5'], ok['fwd_ret'])[0, 1]:.3f}")

print("\n[레이블과 상관이 큰 피처(스피어만, 상위 6)]")
rows = []
for t in ["y_crash", "y_up", "y_logvol"]:
    c = lab[FEATURES + [t]].corr(method="spearman")[t].drop(t).abs().sort_values(ascending=False).head(6)
    rows.append({"레이블": t, **{f"{i + 1}": f"{k}({lab[[k, t]].corr(method='spearman').iloc[0, 1]:+.2f})"
                                for i, k in enumerate(c.index)}})
print(pd.DataFrame(rows).to_string(index=False))
