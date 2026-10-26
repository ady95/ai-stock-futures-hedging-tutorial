"""05-1. 괴리 신호는 몇 번 나오고, 다음 날 시가까지 얼마나 남는가.

선행: chapter04/03_mispricing_features.py
실행: uv run python examples/chapter05/01_signal_persistence.py
"""

import pandas as pd

from hedging.data.collect import load_futures, load_spot
from hedging.strategies.basis_rules import attach_opens, mis_at

pd.set_option("display.width", 160)
for und in ["005930", "000660"]:
    fv = pd.read_parquet(f"data/processed/{und}_features.parquet")
    df = attach_opens(fv, load_futures(und), load_spot(und))
    u = df[df["usable"] & (df["div"] == 0) & (df["dte"] >= 5)]
    rows = []
    for th in [0.2, 0.3, 0.5, 0.75, 1.0]:
        sig = u[u["mis_pct"] >= th].dropna(subset=["next_spot_open", "next_front_open_same"])
        nxt = [mis_at(s, f, r, d, l) for s, f, r, d, l in
               zip(sig["next_spot_open"], sig["next_front_open_same"], sig["rate"], sig["next_date"], sig["ltd"])]
        nxt = pd.Series(nxt, index=sig.index)
        rows.append({"진입 기준%": th, "신호일수": len(sig), "연평균": round(len(sig) / 6.9, 1),
                     "신호일 괴리%": round(sig["mis_pct"].mean(), 3),
                     "다음날 시가 괴리%": round(nxt.mean(), 3),
                     "시가에도 기준 이상": f"{(nxt >= th).mean():.0%}"})
    print(f"=== {und} (분석 사용일·배당 구간 제외·잔존 5거래일 이상)")
    print(pd.DataFrame(rows).to_string(index=False))
