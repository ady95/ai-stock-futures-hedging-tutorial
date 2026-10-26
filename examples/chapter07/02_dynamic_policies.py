"""07-2. VWAP·이동평균·변동성 기반 가변 헤지 정책.

실행: uv run python examples/chapter07/02_dynamic_policies.py
"""

import pandas as pd

from hedging import plotting
from hedging.backtest.hedge_ledger import hedge_metrics, load_hedge_frame, run_hedge
from hedging.strategies.hedge_rules import POLICIES, fixed

pd.set_option("display.width", 220)
paths = {}
for code in ["005930", "000660"]:
    df = load_hedge_frame(code)
    rows = [{"정책": "고정 0.5 (참고)", **hedge_metrics(run_hedge(df, fixed(df, 0.5)))}]
    for name, f in POLICIES.items():
        h = f(df)
        paths[(code, name)] = (df["date"], h, df["spot"])
        res = run_hedge(df, h)
        m = hedge_metrics(res)
        m["h 변경일 비율"] = round(float((h.diff().abs() > 0.005).mean()), 2)
        rows.append({"정책": name, **m})
    print(f"=== {code}")
    print(pd.DataFrame(rows).to_string(index=False))

plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(3, 1, figsize=(10, 7), dpi=150, sharex=True)
d, _, spot = paths[("005930", "추세(60일 이동평균)")]
ax[0].plot(d, spot, lw=1, color="black")
ax[0].set_ylabel("삼성전자 종가")
for name in ["추세(60일 이동평균)", "변동성 목표"]:
    ax[1].plot(d, paths[("005930", name)][1], lw=0.9, label=name)
ax[1].set_ylabel("목표 헤지 비율 h")
ax[1].legend(loc="upper left")
ax[2].plot(d, paths[("005930", "추세+변동성")][1], lw=0.9, color="tab:green", label="추세+변동성")
ax[2].set_ylabel("목표 헤지 비율 h")
ax[2].legend(loc="upper left")
ax[0].set_title("삼성전자: 규칙별 목표 헤지 비율")
fig.tight_layout()
fig.savefig("outputs/ch07_policies_005930.png")
print("그림: outputs/ch07_policies_005930.png")
