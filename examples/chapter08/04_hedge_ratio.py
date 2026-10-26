"""08-3. 예측을 헤지 비율로: 확률·불확실성·비용.

삼성전자·SK하이닉스에서 정책별 h 의 모양(평균·변동·극단 비율), 시드에 따른 h 흔들림,
재조정 밴드(비용) 효과를 본다. 기간은 예측이 있는 2022-01 ~ 데이터 끝.
선행: 03_model_diagnostics.py
실행: uv run python examples/chapter08/04_hedge_ratio.py
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from hedging import plotting
from hedging.backtest.hedge_ledger import HedgeConfig, hedge_metrics, load_hedge_frame, run_hedge
from hedging.strategies.ml_hedge import load_preds, ml_hedge_ratios

pd.set_option("display.width", 220)
START = pd.Timestamp("2022-01-01")
SEEDS = [1, 2, 3, 4, 5]
panel = pd.read_parquet("data/processed/ch08_panel.parquet")
preds = load_preds()


def oos(df, h):
    """예측 기간만 잘라 낸다. 피처가 비어 예측이 없는 날은 전날 h 를 유지한다."""
    m = (df["date"] >= START).values
    return df[m].reset_index(drop=True), h[m].reset_index(drop=True).ffill()


hs = {}
for code, nm in [("005930", "삼성전자"), ("000660", "SK하이닉스")]:
    df = load_hedge_frame(code)
    h = ml_hedge_ratios(df, code, panel, preds)
    hs[code] = (df, h)
    d, ho = oos(df, h)
    print(f"=== {nm}: 정책별 h 의 모양 ({d['date'].min():%Y-%m-%d} ~ {d['date'].max():%Y-%m-%d}, {len(d)}일, "
          f"예측 없어 전날 유지 {int(h[(df['date'] >= START).values]['LGBM 변동성'].isna().sum())}일)")
    rows = []
    for name in ho.columns:
        x = ho[name]
        rows.append({"정책": name, "평균 h": round(x.mean(), 3), "h 표준편차": round(x.std(), 3),
                     "h=0 비율": round((x <= 0).mean(), 3), "h≥0.8 비율": round((x >= 0.8).mean(), 3),
                     "하루 평균 |Δh|": round(x.diff().abs().mean(), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

print("\n[시드에 따른 h 흔들림 — 시드 5개로 각각 만든 h 의 날짜별 표준편차, 중앙값/90% 분위]")
rows = []
for code, (df, _) in hs.items():
    per_seed = [oos(df, ml_hedge_ratios(df, code, panel, preds, seed=s))[1] for s in SEEDS]
    for name in ["LGBM 변동성", "LGBM 상단80%", "LGBM 급락", "LGBM 결합"]:
        sd = pd.concat([p[name] for p in per_seed], axis=1).std(axis=1)
        rows.append({"종목": code, "정책": name, "h 시드 표준편차 중앙값": round(sd.median(), 4),
                     "90% 분위": round(sd.quantile(0.9), 4), "최대": round(sd.max(), 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n[비용: 재조정 밴드 0 vs 3계약 (7-3 권장 범위)]")
rows = []
for code, (df, h) in hs.items():
    d, ho = oos(df, h)
    for name in ho.columns:
        for band in [0, 3]:
            m = hedge_metrics(run_hedge(d, ho[name], replace(HedgeConfig(), band=band)))
            rows.append({"종목": code, "정책": name, "밴드": band, "평균 h": m["평균 h"], "연변동성%": m["연변동성%"],
                         "CVaR5%(일)": m["CVaR5%(일)"], "연 거래 계약": m["연 거래 계약"], "연 비용bp": m["연 비용bp"]})
t = pd.DataFrame(rows)
print(t.pivot_table(index=["종목", "정책"], columns="밴드", values=["연변동성%", "CVaR5%(일)", "연 거래 계약", "연 비용bp"],
                    sort=False).to_string())

plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

df, h = hs["005930"]
d, ho = oos(df, h)
fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=150, sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.6]})
axes[0].plot(d["date"], d["spot"], color="black", lw=0.8)
axes[0].set_ylabel("삼성전자 종가")
for name, color in [("규칙(변동성 목표)", "gray"), ("LGBM 변동성", "tab:blue"), ("LGBM 급락", "tab:red")]:
    axes[1].plot(d["date"], ho[name].rolling(5).mean(), lw=1, color=color, label=name)
axes[1].set_ylabel("헤지 비율 h (5일 평균)")
axes[1].set_ylim(-0.02, 1.02)
axes[1].legend(fontsize=8, loc="upper left")
fig.suptitle("삼성전자: 규칙 기반과 LightGBM 예측 기반 헤지 비율 (2022~)")
fig.tight_layout()
fig.savefig("outputs/ch08_h_005930.png")
print("그림: outputs/ch08_h_005930.png")
