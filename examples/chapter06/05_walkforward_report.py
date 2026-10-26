"""06-4. 워크포워드 검증과 연구 결과 보고서 생성.

후보 규칙 8개(진입 0.3/0.5/0.75/1.0% × 수렴 청산/만기 보유)를 미리 정해 두고,
시험 연도마다 그 이전 거래만으로 규칙을 고른 뒤 시험 연도 성과만 집계한다.

선행: 01_build_universe.py
실행: uv run python examples/chapter06/05_walkforward_report.py
결과: outputs/ch06_research_report.md, outputs/ch06_walkforward.png
"""

from datetime import date
from pathlib import Path

import pandas as pd

from hedging import plotting
from hedging.backtest.report import UNIT, trade_stats
from hedging.backtest.walkforward import candidate_rules, pool_trades, walk_forward
from hedging.data.universe import load_universe
from hedging.strategies.basis_rules import BasisRuleParams

pd.set_option("display.width", 200)
mapping, feats = load_universe()
base = BasisRuleParams(exec="next_open", min_volume=1_000)
rules = candidate_rules(base)
pool = pool_trades(feats, rules)

print("[1] 후보 규칙 전체 기간 성과 (표본 내, 참고용)")
insample = pd.DataFrame({k: trade_stats(g) for k, g in pool.groupby("rule")}).T
print(insample.to_string())

years = [2022, 2023, 2024, 2025, 2026]
wf, oos = walk_forward(pool, years, min_trades=20)
print("\n[2] 워크포워드 (확장 학습 창, 시험 연도별)")
print(wf.to_string(index=False))
print("\n[3] 표본 밖 합계:", trade_stats(oos))

# 규칙 선택 없이 '학습 최고 규칙'을 그대로 썼다면 (양수 조건 무시)
naive = pd.concat([pool[(pool["entry_date"].dt.year == r["시험연도"]) & (pool["rule"] == r["학습 최고 규칙"])]
                   for _, r in wf.iterrows()])
print("    학습 최고 규칙을 무조건 썼다면:", trade_stats(naive))

# 강건성 점검: 표본 밖 이익은 어디서 왔나
print("\n[4] 강건성 점검")
o = oos.assign(krw=oos["net_bp"] * UNIT / 1e4)
by_month = o.groupby(o["entry_date"].dt.to_period("M"))["krw"].agg(["size", "sum"])
by_day = o.groupby("entry_date")["krw"].sum().sort_values(ascending=False)
srt = o["krw"].sort_values(ascending=False).values
top5 = by_day.head(5)
robust = {
    "진입일 상위 5일 비중": round(float(top5.sum() / o["krw"].sum()), 2),
    "상위 10% 거래 비중": round(float(srt[: len(srt) // 10].sum() / srt.sum()), 2),
    "상위 5일 제외 평균bp": round(float(o[~o["entry_date"].isin(top5.index)]["net_bp"].mean()), 1),
    "거래 종목 수": o["und_code"].nunique(),
    "종목 합계가 양수인 종목 비율": round(float((o.groupby("und_code")["krw"].sum() > 0).mean()), 2),
}
print(" ", robust)
print("  월별(만원):", {str(k): int(v / 1e4) for k, v in by_month["sum"].items()})
slip1 = pool_trades(feats, {r: rules[r].__class__(**{**rules[r].__dict__, "slip_ticks": 1.0})
                            for r in wf["선택 규칙"].unique() if r != "무거래"})
s1 = pd.concat([slip1[(slip1["rule"] == r["선택 규칙"]) & (slip1["entry_date"].dt.year == r["시험연도"])]
                for _, r in wf.iterrows() if r["선택 규칙"] != "무거래"])
print("  같은 선택, 슬리피지 1틱:", trade_stats(s1),
      {int(y): round(float(g["net_bp"].mean()), 1) for y, g in s1.groupby(s1["entry_date"].dt.year)})

# 그림: 표본 밖 누적 손익(1억 환산)
plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

Path("outputs").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
for name, t in [("워크포워드 선택", oos), ("학습 최고 규칙 무조건", naive)]:
    if len(t):
        s = t.sort_values("exit_date")
        ax.step(s["exit_date"], (s["net_bp"] * UNIT / 1e4).cumsum() / 1e4, where="post", label=name)
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("누적 순손익(만원, 거래당 1억 환산)")
ax.set_title("표본 밖(2022~2026) 누적 순손익 — 다음 날 시가 체결")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig("outputs/ch06_walkforward.png")

# 연구 결과 보고서 (Markdown)
n_stocks = len(feats)
lines = [
    "# 베이시스 전략 연구 결과 보고서",
    "",
    f"- 생성일: {date.today()}  ·  예제코드 태그: ch06",
    f"- 대상: KRX 주식선물 기초자산 중 현물 종가 검증을 통과한 {n_stocks}종목 (상장 기간표로 그날 상장된 종목만)",
    f"- 기간: {min(f['date'].min() for f in feats.values()):%Y-%m-%d} ~ {max(f['date'].max() for f in feats.values()):%Y-%m-%d}",
    "- 전략: 선물 괴리 ≥ 기준이면 현물 매수 + 선물 매도, 다음 날 시가 체결, 비용(수수료·매도세 0.20%·슬리피지 반 틱·자금비용) 차감",
    "",
    "## 가설",
    "",
    "H1: 비용을 초과한 이론 베이시스 괴리의 일부는 이후 수렴한다. (반증: 비용 차감 후 이익 없음)",
    "",
    "## 후보 규칙 전체 기간 성과 (표본 내, 참고)",
    "",
    insample.to_markdown(),
    "",
    "## 워크포워드 (확장 학습 창)",
    "",
    wf.to_markdown(index=False),
    "",
    f"표본 밖 합계: {trade_stats(oos)}",
    "",
    "## 강건성 점검",
    "",
    f"- {robust}",
    f"- 월별 손익(만원): { {str(k): int(v / 1e4) for k, v in by_month['sum'].items()} }",
    f"- 같은 선택에 슬리피지 1틱: {trade_stats(s1)}",
    "",
    "## 판정",
    "",
    ("표본 밖 평균은 양수지만, 학습 근거가 약하고(학습 평균 수 bp) 이익이 특정 시기에 몰려 있다. "
     "강건성 점검 결과와 함께 판단해야 하며, 단일 시기 의존이면 H1은 '판정 유보'로 둔다."
     if len(oos) and oos["net_bp"].mean() > 0 else
     "표본 밖에서 비용 차감 후 이익을 확인하지 못했다. 현실적 체결 가정(다음 날 시가)에서 H1은 지지되지 않는다."),
    "",
    "## 한계",
    "",
    "- 일봉 종가·시가만 사용. 장중 동시 체결 가능성은 검증하지 않았다(12장 실시간 기록으로 보완 예정)",
    "- 슬리피지는 호가단위 기반 가정이며, 거래가 적은 종목은 실제 스프레드가 더 넓을 수 있다",
    "- 현물 코드 매핑에 실패한 종목은 제외했다(매핑 표 참조)",
    "- 세금·수수료는 예시 값이며 개인별로 다르다",
]
Path("outputs/ch06_research_report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n보고서: outputs/ch06_research_report.md, 그림: outputs/ch06_walkforward.png")
