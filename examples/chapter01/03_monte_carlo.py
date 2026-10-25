"""01-4. 실습: 가격경로별 롱·숏 손익 시뮬레이터.

무작위 경로 10,000개에서 세 방식(동시 청산 / 분리 청산 / 사후 최적)의 손익 분포를 비교한다.
실행: uv run python examples/chapter01/03_monte_carlo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hedging import plotting
from hedging.pnl import CostModel, hindsight_pnl, hold_both_pnl, split_exit_pnl
from hedging.sim import gbm_paths

S0, QTY, DAYS = 100_000, 100, 60
SIGMA, MU, N_PATHS, SEED = 0.30, 0.0, 10_000, 42
cost = CostModel()

paths = gbm_paths(S0, MU, SIGMA, DAYS, N_PATHS, seed=SEED)


def summarize(name: str, pnl: np.ndarray) -> dict:
    return {
        "방식": name,
        "평균": pnl.mean(),
        "중앙값": np.median(pnl),
        "승률": (pnl > 0).mean(),
        "하위 5%": np.quantile(pnl, 0.05),
        "최악": pnl.min(),
        "최선": pnl.max(),
    }


both = np.array([hold_both_pnl(S0, p[-1], QTY, DAYS, cost).net for p in paths])
split_res = [split_exit_pnl(p, QTY, 0.05, cost) for p in paths]
split = np.array([r.net for r in split_res])
hind = np.array([hindsight_pnl(p, QTY, cost).net for p in paths])

table = pd.DataFrame([
    summarize("동시 청산", both),
    summarize("분리 청산(5%)", split),
    summarize("사후 최적(불가능)", hind),
])
money = "{:,.0f}".format
print(f"경로 {N_PATHS:,}개 · {DAYS}거래일 · 연변동성 {SIGMA:.0%} · 기대수익 {MU:.0%} · seed {SEED} · 비용 포함")
print(table.to_string(index=False, formatters={
    "평균": money, "중앙값": money, "승률": "{:.1%}".format,
    "하위 5%": money, "최악": money, "최선": money}))

# 분리 청산의 손익을 '어떻게 끝났는가'로 나눠 본다
first = np.array([r.first_leg is not None for r in split_res])
recovered = np.array([r.recovered for r in split_res])
gross = np.array([r.gross for r in split_res])
print()
print(f"분리 청산의 비용 제외 평균: {gross.mean():,.0f}원 (경로 1개당 가격손익의 평균)")
print()
print("분리 청산 경로 분해")
for label, mask in [
    ("목표 미도달(동시 청산)", ~first),
    ("한쪽 청산 → 본전 복귀", recovered),
    ("한쪽 청산 → 만기까지 미복귀", first & ~recovered),
]:
    print(f"  {label:<20} {mask.mean():6.1%}  평균 {split[mask].mean():>12,.0f}원")

# 목표수익을 바꿔도 평균은 0 근처 - 비용에 머문다
print()
print("목표수익별 분리 청산 (비용 포함)")
for tp in [0.03, 0.05, 0.10]:
    pnl = np.array([split_exit_pnl(p, QTY, tp, cost).net for p in paths])
    print(f"  목표 {tp:>4.0%}: 평균 {pnl.mean():>9,.0f}원  승률 {(pnl > 0).mean():6.1%}  하위5% {np.quantile(pnl, 0.05):>12,.0f}원")

# 히스토그램 저장
plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

out = Path("outputs")
out.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
man = 10_000  # 만원 단위
bins = np.linspace(-300, 300, 121)
ax.hist(np.clip(split / man, bins[0], bins[-1]), bins=bins, color="tab:orange", alpha=0.8,
        label="분리 청산(5%) 경로별 순손익")
ax.axvline(split.mean() / man, color="tab:red", lw=1.5, ls="--", label=f"분리 청산 평균 {split.mean() / man:,.1f}만원")
ax.axvline(both.mean() / man, color="tab:blue", lw=1.5, label=f"동시 청산 평균 {both.mean() / man:,.1f}만원")
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("순손익(만원) — 왼쪽 끝 막대는 -300만원보다 큰 손실을 모은 것")
ax.set_ylabel("경로 수")
ax.set_title("롱·숏 각 100주, 60거래일, 무작위 경로 10,000개 (비용 포함)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(out / "ch01_split_exit_hist.png")
print(f"\n그림 저장: {out / 'ch01_split_exit_hist.png'}")
