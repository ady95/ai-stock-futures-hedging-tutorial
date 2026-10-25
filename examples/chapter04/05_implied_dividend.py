"""04-2. 선물 가격에서 시장이 예상하는 배당을 역산한다.

    시장 내재 배당 = S × (1 + r × τ) − F

선행: 03_mispricing_features.py
실행: uv run python examples/chapter04/05_implied_dividend.py [005930] [2026-09-08] [2026-10-08]
"""

import sys

import pandas as pd

und = sys.argv[1] if len(sys.argv) > 1 else "005930"
start = sys.argv[2] if len(sys.argv) > 2 else "2026-09-08"
end = sys.argv[3] if len(sys.argv) > 3 else "2026-10-08"

fv = pd.read_parquet(f"data/processed/{und}_features.parquet")
w = fv[(fv["date"] >= start) & (fv["date"] <= end)].copy()
w["implied_div"] = w["spot"] * (1 + w["rate"] * w["tau"]) - w["front_close"]
view = w[["date", "spot", "front_code", "front_close", "div", "implied_div", "mis_pct", "cal_dte"]].rename(
    columns={"div": "예상배당", "implied_div": "내재배당", "mis_pct": "괴리율%", "cal_dte": "잔존일"})
view["date"] = view["date"].dt.strftime("%Y-%m-%d")
print(view.to_string(index=False, formatters={"내재배당": "{:,.0f}".format, "괴리율%": "{:+.2f}".format}))
