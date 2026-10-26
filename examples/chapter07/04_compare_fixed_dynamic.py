"""07-4. 고정 헤지와 가변 헤지 비교 — 가설 H2.

동적 규칙은 '평균 헤지 비율이 같은 고정 헤지'와 비교한다(고정 h 0~1 곡선에서 보간).
실행: uv run python examples/chapter07/04_compare_fixed_dynamic.py
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from hedging import plotting
from hedging.backtest.hedge_ledger import HedgeConfig, hedge_metrics, load_hedge_frame, run_hedge
from hedging.data.universe import load_universe
from hedging.strategies.hedge_rules import POLICIES, fixed

pd.set_option("display.width", 230)
GRID = np.round(np.arange(0, 1.01, 0.1), 2)
KEYS = ["연수익%", "연변동성%", "최대낙폭%", "CVaR5%(일)", "최악일%"]


def frontier(df, cfg=HedgeConfig()):
    return pd.DataFrame([{"h": h, **hedge_metrics(run_hedge(df, fixed(df, h), cfg))} for h in GRID])


def vs_fixed(m, fr):
    """같은 평균 h 의 고정 헤지(보간) 대비 차이. 양수 = 동적 규칙이 더 큼."""
    out = {}
    for k in KEYS:
        out[k] = round(m[k] - float(np.interp(m["평균 h"], fr["h"], fr[k])), 2)
    # 같은 변동성에서의 수익 차이 (수익-변동성 효율)
    order = fr.sort_values("연변동성%")
    out["같은 변동성 수익 차"] = round(m["연수익%"] - float(np.interp(m["연변동성%"], order["연변동성%"], order["연수익%"])), 2)
    return out


fronts, points = {}, {}
for code in ["005930", "000660"]:
    df = load_hedge_frame(code)
    fr = frontier(df)
    fronts[code] = fr
    rows = []
    for name, f in POLICIES.items():
        m = hedge_metrics(run_hedge(df, f(df)))
        points[(code, name)] = m
        rows.append({"정책": name, "평균 h": m["평균 h"], **vs_fixed(m, fr)})
    print(f"=== {code}: 같은 평균 h 고정 헤지 대비 차이 (동적 − 고정. 낙폭·CVaR·최악일은 음수 값이므로 차이가 양수면 개선, 변동성은 음수면 개선)")
    print(pd.DataFrame(rows).to_string(index=False))

print("\n[시험 연도별, 삼성전자: 추세+변동성 − 같은 평균 h 고정]")
df = load_hedge_frame("005930")
h = POLICIES["추세+변동성"](df)
rows = []
for y in range(2020, 2027):
    sub = df[df["date"].dt.year == y].reset_index(drop=True)
    hy = h[df["date"].dt.year == y].reset_index(drop=True)
    m = hedge_metrics(run_hedge(sub, hy))
    fr = frontier(sub)
    rows.append({"연도": y, "평균 h": m["평균 h"], **vs_fixed(m, fr)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n[하루 지연 실행 민감도: 같은 변동성 수익 차, 연변동성 차]")
rows = []
for code in ["005930", "000660"]:
    df = load_hedge_frame(code)
    for name, f in POLICIES.items():
        for ex in ["close", "lag1"]:
            cfg = replace(HedgeConfig(), exec=ex)
            m = hedge_metrics(run_hedge(df, f(df), cfg))
            d = vs_fixed(m, fronts[code])
            rows.append({"종목": code, "정책": name, "체결": ex, "같은 변동성 수익 차": d["같은 변동성 수익 차"],
                         "CVaR 차": d["CVaR5%(일)"], "최대낙폭 차": d["최대낙폭%"]})
print(pd.DataFrame(rows).pivot_table(index=["종목", "정책"], columns="체결",
                                     values=["같은 변동성 수익 차", "CVaR 차", "최대낙폭 차"]).to_string())

print("\n[유동성 상위 20종목 강건성: 전 기간 상장 종목 중 최근월물 평균 거래량 상위]")
mapping, feats = load_universe()
full = {c: f for c, f in feats.items() if f["date"].min() <= pd.Timestamp("2019-12-31")
        and f["date"].max() == max(x["date"].max() for x in feats.values())}
top = sorted(full, key=lambda c: full[c]["front_volume"].mean(), reverse=True)[:20]
rows = []
for code in top:
    df = load_hedge_frame(code)
    fr = frontier(df)
    for name, f in POLICIES.items():
        m = hedge_metrics(run_hedge(df, f(df)))
        rows.append({"종목": code, "정책": name, **vs_fixed(m, fr)})
rb = pd.DataFrame(rows)
summary = rb.groupby("정책").agg(
    같은변동성수익차_중앙=("같은 변동성 수익 차", "median"),
    수익효율_개선비율=("같은 변동성 수익 차", lambda x: (x > 0).mean()),
    변동성차_중앙=("연변동성%", "median"),
    CVaR차_중앙=("CVaR5%(일)", "median"),
    CVaR_개선비율=("CVaR5%(일)", lambda x: (x > 0).mean()),
    최대낙폭차_중앙=("최대낙폭%", "median"),
    최대낙폭_개선비율=("최대낙폭%", lambda x: (x > 0).mean()),
)
print(summary.round(2).to_string())
print("대상:", ", ".join(mapping.drop_duplicates("code").set_index("code").loc[top, "und_name"]))

plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
for ax, code, nm in [(axes[0], "005930", "삼성전자"), (axes[1], "000660", "SK하이닉스")]:
    fr = fronts[code]
    ax.plot(fr["연변동성%"], fr["연수익%"], "-o", ms=3, color="gray", label="고정 헤지 h=0~1")
    for h0 in [0.0, 0.5, 1.0]:
        r = fr[fr["h"] == h0].iloc[0]
        ax.annotate(f"h={h0}", (r["연변동성%"], r["연수익%"]), fontsize=7, xytext=(4, -10), textcoords="offset points")
    for name in POLICIES:
        m = points[(code, name)]
        ax.scatter(m["연변동성%"], m["연수익%"], s=30, label=name, zorder=3)
    ax.set_xlabel("연변동성 %")
    ax.set_ylabel("연수익 %")
    ax.set_title(f"{nm}: 고정 헤지 곡선과 동적 규칙")
    ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig("outputs/ch07_frontier.png")
print("그림: outputs/ch07_frontier.png")
