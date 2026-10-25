"""04-4. 베이시스·괴리 시각화 — Plotly 대화형 대시보드 + 정적 그림.

선행: 03_mispricing_features.py
실행: uv run python examples/chapter04/04_basis_dashboard.py [005930]
결과: outputs/ch04_dashboard_<코드>.html (브라우저로 열기), outputs/ch04_basis_<코드>.png
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hedging import plotting

und = sys.argv[1] if len(sys.argv) > 1 else "005930"
fv = pd.read_parquet(f"data/processed/{und}_features.parquet")
out = Path("outputs")
out.mkdir(exist_ok=True)

# ---- 1) Plotly 대화형 대시보드 ----------------------------------------------
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.3, 0.25, 0.25, 0.2],
                    subplot_titles=("현물·최근월물 종가", "실제 베이시스 vs 이론 베이시스 (%)",
                                    "괴리율 (%) — 분석 사용일만", "최근월물 거래량"))
fig.add_trace(go.Scatter(x=fv.date, y=fv.spot, name="현물", line=dict(width=1)), 1, 1)
fig.add_trace(go.Scatter(x=fv.date, y=fv.front_close, name="최근월물", line=dict(width=1)), 1, 1)
fig.add_trace(go.Scatter(x=fv.date, y=fv.basis_pct, name="실제 베이시스", line=dict(width=1)), 2, 1)
fig.add_trace(go.Scatter(x=fv.date, y=fv.theo_basis_pct, name="이론 베이시스", line=dict(width=1.5)), 2, 1)
u = fv[fv.usable]
fig.add_trace(go.Scatter(x=u.date, y=u.mis_pct, name="괴리율", mode="markers", marker=dict(size=3)), 3, 1)
for y in (-0.3, 0.3):
    fig.add_hline(y=y, line_dash="dot", line_color="gray", row=3, col=1)
fig.add_trace(go.Bar(x=fv.date, y=fv.front_volume, name="거래량"), 4, 1)
fig.update_layout(height=1000, title=f"{und} 현물–주식선물 베이시스 대시보드", hovermode="x unified")
fig.write_html(out / f"ch04_dashboard_{und}.html", include_plotlyjs="cdn")

# ---- 2) 책에 넣을 정적 그림 (최근 1년) --------------------------------------
plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

recent = fv[fv.date >= fv.date.max() - pd.Timedelta(days=365)]
ru = recent[recent.usable]
f, ax = plt.subplots(3, 1, figsize=(9, 7.5), dpi=150, sharex=True,
                     gridspec_kw={"height_ratios": [1.2, 1, 1]})
ax[0].plot(recent.date, recent.spot, lw=1, label="현물")
ax[0].plot(recent.date, recent.front_close, lw=1, label="최근월물")
ax[0].set_ylabel("가격(원)")
ax[0].legend(loc="upper left")
ax[1].plot(recent.date, recent.basis_pct, lw=0.8, color="tab:gray", label="실제 베이시스")
ax[1].plot(recent.date, recent.theo_basis_pct, lw=1.5, color="tab:red", label="이론 베이시스")
ax[1].set_ylabel("현물 대비 %")
ax[1].legend(loc="upper left")
ax[2].scatter(ru.date, ru.mis_pct, s=6, color="tab:blue", label="괴리율(분석 사용일)")
for y in (-0.3, 0.3):
    ax[2].axhline(y, ls=":", color="gray", lw=0.8)
ax[2].axhline(0, color="black", lw=0.6)
ax[2].set_ylabel("괴리율 %")
ax[2].legend(loc="upper left")
ax[0].set_title(f"{und} 최근 1년: 현물·선물, 베이시스, 괴리 (점선 ±0.3%)")
f.tight_layout()
f.savefig(out / f"ch04_basis_{und}.png")
print(f"저장: {out / f'ch04_dashboard_{und}.html'}, {out / f'ch04_basis_{und}.png'}")
print(f"최근 1년 {len(recent)}일, 분석 사용 {len(ru)}일, |괴리|>0.3% {int((ru.mis_pct.abs() > 0.3).sum())}일")
