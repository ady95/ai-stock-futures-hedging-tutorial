"""05-2. 현물 매수 + 선물 매도 2-leg 진입·청산 — 거래 목록 만들기.

실행: uv run python examples/chapter05/02_two_leg_trades.py [005930] [진입%=0.5]
"""

import sys

import pandas as pd

from hedging.data.collect import load_futures, load_spot
from hedging.strategies.basis_rules import BasisRuleParams, attach_opens, run, summarize

und = sys.argv[1] if len(sys.argv) > 1 else "005930"
entry = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
fv = pd.read_parquet(f"data/processed/{und}_features.parquet")
df = attach_opens(fv, load_futures(und), load_spot(und))

p = BasisRuleParams(entry_pct=entry, exec="next_open")
t = run(df, p)
print(f"{und} 진입 괴리 {entry}% · 다음 날 시가 체결 · 현물 {p.qty:,}주 + 선물 {p.qty // 10}계약")
print({k: round(v, 3) if isinstance(v, float) else v for k, v in summarize(t).items()})
cols = ["code", "signal_date", "entry_date", "exit_date", "reason", "days",
        "signal_mis_pct", "entry_mis_pct", "exit_mis_pct", "gross", "net", "net_bp"]
view = t[cols].tail(8).copy()
for c in ("signal_date", "entry_date", "exit_date"):
    view[c] = view[c].dt.strftime("%Y-%m-%d")
print(view.round(3).to_string(index=False))
