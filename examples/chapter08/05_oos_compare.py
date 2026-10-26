"""08-4. 규칙 기반 헤지 대비 외표본 성과 비교 — 가설 H3.

H3: LightGBM은 규칙 기반 헤지의 위험조정 성과를 개선한다(독립 외표본에서 효과 없으면 기각).
기간: 2022-01 ~ 데이터 끝(워크포워드 시험 구간), 대상: 유동성 상위 20종목(삼성전자·SK하이닉스 포함).
각 정책은 (1) 같은 평균 h 의 고정 헤지, (2) 7장 규칙(변동성 목표)과 비교한다.
LightGBM 정책은 시드 5개 각각으로 장부를 돌려 평균과 편차를 낸다.
선행: 03_model_diagnostics.py
실행: uv run python examples/chapter08/05_oos_compare.py   (수 분 걸린다)
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from hedging import plotting
from hedging.backtest.hedge_ledger import (HedgeConfig, fixed_frontier, hedge_metrics, load_hedge_frame, run_hedge,
                                           vs_fixed)
from hedging.data.universe import load_universe
from hedging.models.lgbm_hedge import top_liquid_codes
from hedging.strategies.ml_hedge import load_preds, ml_hedge_ratios

pd.set_option("display.width", 230)
START = pd.Timestamp("2022-01-01")
SEEDS = [1, 2, 3, 4, 5]
RULE = "규칙(변동성 목표)"
LGBM = ["LGBM 변동성", "LGBM 상단80%", "LGBM 급락", "LGBM 결합"]
panel = pd.read_parquet("data/processed/ch08_panel.parquet")
preds = load_preds()
mapping, feats = load_universe()
codes = top_liquid_codes(feats, 20)
names = mapping.drop_duplicates("code").set_index("code")["und_name"]


def oos(df, h):
    m = (df["date"] >= START).values
    return df[m].reset_index(drop=True), h[m].reset_index(drop=True).ffill()


def evaluate(d, ho, fr):
    """정책별 지표, 고정 헤지 대비 차이, 규칙 대비 차이."""
    res = {name: hedge_metrics(run_hedge(d, ho[name])) for name in ho.columns}
    rows = []
    for name, m in res.items():
        r = res[RULE]
        rows.append({"정책": name, **{f"m_{k}": v for k, v in m.items()}, **vs_fixed(m, fr),
                     "규칙 대비 연변동성": round(m["연변동성%"] - r["연변동성%"], 2),
                     "규칙 대비 CVaR": round(m["CVaR5%(일)"] - r["CVaR5%(일)"], 3),
                     "규칙 대비 최대낙폭": round(m["최대낙폭%"] - r["최대낙폭%"], 2),
                     "규칙 대비 비용bp": round(m["연 비용bp"] - r["연 비용bp"], 1)})
    return rows


rows, seed_rows, fronts, points = [], [], {}, {}
for code in codes:
    df = load_hedge_frame(code)
    d, ho = oos(df, ml_hedge_ratios(df, code, panel, preds))
    fr = fixed_frontier(d)
    fronts[code] = fr
    for r in evaluate(d, ho, fr):
        rows.append({"종목": code, **r})
    for s in SEEDS:
        _, hs = oos(df, ml_hedge_ratios(df, code, panel, preds, seed=s))
        hs[RULE] = ho[RULE]
        for r in evaluate(d, hs[[RULE] + LGBM], fr):
            if r["정책"] != RULE:
                seed_rows.append({"종목": code, "seed": s, **r})
res = pd.DataFrame(rows)

for code, nm in [("005930", "삼성전자"), ("000660", "SK하이닉스")]:
    t = res[res["종목"] == code]
    print(f"=== {nm} (2022-01 ~ 2026-10, 시드 평균 예측)")
    out = t[["정책", "m_평균 h", "m_연수익%", "m_연변동성%", "m_최대낙폭%", "m_CVaR5%(일)", "m_연 비용bp",
             "같은 변동성 수익 차", "CVaR5%(일)", "최대낙폭%"]].copy()
    out.columns = ["정책", "평균 h", "연수익%", "연변동성%", "최대낙폭%", "CVaR5%(일)", "연 비용bp",
                   "같은 변동성 수익 차", "CVaR 차(고정 대비)", "낙폭 차(고정 대비)"]
    print(out.to_string(index=False))

print("\n[20종목 요약 — 같은 평균 h 고정 헤지 대비 (양수 = 개선; 변동성은 음수 = 개선)]")
summ = res.groupby("정책", sort=False).agg(
    평균h_중앙=("m_평균 h", "median"),
    같은변동성수익차_중앙=("같은 변동성 수익 차", "median"),
    수익효율_개선비율=("같은 변동성 수익 차", lambda x: (x > 0).mean()),
    CVaR차_중앙=("CVaR5%(일)", "median"),
    CVaR_개선비율=("CVaR5%(일)", lambda x: (x > 0).mean()),
    낙폭차_중앙=("최대낙폭%", "median"),
    낙폭_개선비율=("최대낙폭%", lambda x: (x > 0).mean()),
)
print(summ.round(3).to_string())

print("\n[20종목 요약 — 7장 규칙(변동성 목표) 대비 (CVaR·낙폭은 양수 = 규칙보다 개선)]")
summ2 = res[res["정책"] != RULE].groupby("정책", sort=False).agg(
    연변동성차_중앙=("규칙 대비 연변동성", "median"),
    CVaR차_중앙=("규칙 대비 CVaR", "median"),
    CVaR_규칙보다나은종목=("규칙 대비 CVaR", lambda x: int((x > 0).sum())),
    낙폭차_중앙=("규칙 대비 최대낙폭", "median"),
    비용차_중앙bp=("규칙 대비 비용bp", "median"),
)
print(summ2.round(3).to_string())

print("\n[시드 5개 각각의 20종목 중앙값 — 평균 ± 표준편차]")
sr = pd.DataFrame(seed_rows)
per_seed = sr.groupby(["정책", "seed"], sort=False).agg(
    같은변동성수익차=("같은 변동성 수익 차", "median"), CVaR차_고정=("CVaR5%(일)", "median"),
    CVaR차_규칙=("규칙 대비 CVaR", "median"), 규칙보다나은종목=("규칙 대비 CVaR", lambda x: (x > 0).sum()))
agg = per_seed.groupby("정책", sort=False).agg(["mean", "std"])
print(agg.round(3).to_string())

print("\n[연도별: 20종목 중앙값, 같은 평균 h 고정 대비 CVaR 차]")
yrows = []
for code in codes:
    df = load_hedge_frame(code)
    d, ho = oos(df, ml_hedge_ratios(df, code, panel, preds))
    for y in sorted(d["date"].dt.year.unique()):
        k = (d["date"].dt.year == y).values
        dy, hy = d[k].reset_index(drop=True), ho[k].reset_index(drop=True)
        fy = fixed_frontier(dy)
        for name in [RULE, "HAR 변동성", "LGBM 변동성", "LGBM 상단80%"]:
            yrows.append({"연도": y, "정책": name, **vs_fixed(hedge_metrics(run_hedge(dy, hy[name])), fy)})
yr = pd.DataFrame(yrows)
print(yr.pivot_table(index="연도", columns="정책", values="CVaR5%(일)", aggfunc="median", sort=False).round(3).to_string())

print("\n[하루 지연 실행 민감도 — 같은 평균 h 고정 대비 CVaR 차]")
lrows = []
for code in ["005930", "000660"]:
    df = load_hedge_frame(code)
    d, ho = oos(df, ml_hedge_ratios(df, code, panel, preds))
    for ex in ["close", "lag1"]:
        cfg = replace(HedgeConfig(), exec=ex)
        fr = fixed_frontier(d, cfg)
        for name in ho.columns:
            m = hedge_metrics(run_hedge(d, ho[name], cfg))
            lrows.append({"종목": code, "정책": name, "체결": ex, "CVaR 차": vs_fixed(m, fr)["CVaR5%(일)"]})
print(pd.DataFrame(lrows).pivot_table(index=["종목", "정책"], columns="체결", values="CVaR 차", sort=False).to_string())
print("대상:", ", ".join(names.loc[codes]))

plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
for ax, code, nm in [(axes[0], "005930", "삼성전자"), (axes[1], "000660", "SK하이닉스")]:
    fr = fronts[code]
    ax.plot(fr["연변동성%"], fr["연수익%"], "-o", ms=3, color="gray", label="고정 헤지 h=0~1")
    for h0 in [0.0, 0.5, 1.0]:
        r = fr[fr["h"] == h0].iloc[0]
        ax.annotate(f"h={h0}", (r["연변동성%"], r["연수익%"]), fontsize=7, xytext=(4, -10), textcoords="offset points")
    t = res[res["종목"] == code]
    for _, r in t.iterrows():
        ax.scatter(r["m_연변동성%"], r["m_연수익%"], s=30, label=r["정책"], zorder=3,
                   marker="s" if r["정책"] == RULE else "o")
    ax.set_xlabel("연변동성 %")
    ax.set_ylabel("연수익 %")
    ax.set_title(f"{nm} 2022~: 고정 헤지 곡선과 정책")
    ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig("outputs/ch08_oos_frontier.png")
print("그림: outputs/ch08_oos_frontier.png")
