"""06-1. 현물·선물 통합 장부 백테스트 — 한 종목.

선행: 01_build_universe.py (data/processed/universe/<코드>_features.parquet)
실행: uv run python examples/chapter06/02_ledger_backtest.py [005930]
"""

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from hedging import plotting
from hedging.backtest.ledger import LedgerConfig, cash_benchmark, run_ledger
from hedging.backtest.report import ledger_metrics, trade_stats
from hedging.strategies.basis_rules import BasisRuleParams

pd.set_option("display.width", 160)
code = sys.argv[1] if len(sys.argv) > 1 else "005930"
df = pd.read_parquet(f"data/processed/universe/{code}_features.parquet")
cfg = LedgerConfig(capital=200_000_000, target_notional=100_000_000)
base = BasisRuleParams(entry_pct=0.5, exec="next_open")

results = {}
for ex in ["close", "next_open"]:
    led, tr = run_ledger(df, replace(base, exec=ex), cfg)
    bench = cash_benchmark(led, df, cfg.capital)
    results[ex] = (led, tr, bench)
    m = ledger_metrics(led, bench)
    print(f"=== {code} 체결={ex} (자본 2억, 진입 1억, 진입 괴리 0.5%)")
    for k, v in m.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  거래:", trade_stats(tr))

led, tr, bench = results["next_open"]
last = led.iloc[-1]
print("\n손익 분해 (다음 날 시가 체결, 누적, 원)")
for k in ["spot_pnl", "fut_pnl", "costs", "interest"]:
    print(f"  {k:<9} {last[k]:>15,.0f}")
print(f"  현물+선물 {last['spot_pnl'] + last['fut_pnl']:>15,.0f}  (베이시스 손익)")

view = tr[["code", "entry_date", "exit_date", "reason", "qty", "spot_pnl", "fut_pnl", "costs", "net", "net_bp"]].tail(5)
print("\n최근 거래 5건\n" + view.assign(entry_date=view.entry_date.dt.strftime("%Y-%m-%d"),
                                    exit_date=view.exit_date.dt.strftime("%Y-%m-%d")).round(1).to_string(index=False))

plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

Path("outputs").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
for ex, (lg, _, bm) in results.items():
    ax.plot(lg["date"], (lg["equity"].values - bm.values) / 1e4, lw=1.2,
            label=f"전략 - 현금 보유 ({'종가 체결' if ex == 'close' else '다음 날 시가 체결'})")
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("초과 평가액(만원)")
ax.set_title(f"{code} 베이시스 전략: 현금 보유 대비 누적 초과 손익 (자본 2억)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"outputs/ch06_ledger_{code}.png")
print(f"\n그림 저장: outputs/ch06_ledger_{code}.png")
