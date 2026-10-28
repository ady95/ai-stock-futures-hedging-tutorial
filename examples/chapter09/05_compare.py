"""09-4. PPO 대 LightGBM·규칙 기반 외표본 비교 — 가설 H4.

H4: PPO는 비용·증거금·체결 위험을 반영한 순차 의사결정에서 추가 개선을 제공한다(불안정하면 기각).
기간: 2022-01-03 ~ 2026-10-22(마지막 날은 다음 날 손익이 없어 제외), 삼성전자·SK하이닉스.
선행: 03_train.py (PPO), 선택: 03_train.py --algo sac (SAC 강건성 표), examples/chapter08/03_model_diagnostics.py
실행: uv run python examples/chapter09/05_compare.py
"""

import glob
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hedging import plotting
from hedging.backtest.hedge_ledger import (HedgeConfig, fixed_frontier, hedge_metrics, load_hedge_frame, run_hedge,
                                           vs_fixed)
from hedging.instruments.contracts import tick_size
from hedging.rl.train import load_frames, run_policy
from hedging.strategies.ml_hedge import load_preds, ml_hedge_ratios

pd.set_option("display.width", 230)
YEARS = [2022, 2023, 2024, 2025, 2026]
SEEDS = [1, 2, 3, 4, 5]
CODES = {"005930": "삼성전자", "000660": "SK하이닉스"}
MODELS = "outputs/ch09_models"
panel = pd.read_parquet("data/processed/ch08_panel.parquet")
preds = load_preds()
frames = load_frames(panel)
KEYS = ["평균 h", "연수익%", "연변동성%", "최대낙폭%", "CVaR5%(일)", "연 비용bp", "연 거래 계약"]


def policy_h(code, algo):
    """시드별 h 시계열(시험 구간). 반환: {seed: DataFrame(date, h, want)}."""
    out = {}
    for s in SEEDS:
        paths = {y: f"{MODELS}/{algo}_{y}_s{s}" for y in YEARS}
        if all(Path(p + ".zip").exists() for p in paths.values()):
            out[s] = run_policy(frames, code, YEARS, paths, algo)
    return out


def row(name, m, fr, seed=None):
    return {"정책": name, "seed": seed, **{k: m[k] for k in KEYS}, **{f"고정 대비 {k}": v for k, v in vs_fixed(m, fr).items()}}


rows, hs_all, ds = [], {}, {}
for code, nm in CODES.items():
    df = load_hedge_frame(code)
    ppo = policy_h(code, "ppo")
    sac = policy_h(code, "sac")
    dates = ppo[SEEDS[0]]["date"]
    m = df["date"].isin(dates).values
    d = df[m].reset_index(drop=True)
    ml = ml_hedge_ratios(df, code, panel, preds)[m].reset_index(drop=True).ffill()
    fr = fixed_frontier(d)
    ds[code] = (d, fr)
    hs = {"규칙(변동성 목표)": ml["규칙(변동성 목표)"], "LGBM 변동성": ml["LGBM 변동성"]}
    for name, h in hs.items():
        rows.append({"종목": code, **row(name, hedge_metrics(run_hedge(d, h)), fr)})
    for algo, pol in [("PPO", ppo), ("SAC", sac)]:
        for s, p in pol.items():
            h = p["h"].reset_index(drop=True)
            hs[f"{algo} s{s}"] = h
            rows.append({"종목": code, **row(algo, hedge_metrics(run_hedge(d, h)), fr, s)})
    hs_all[code] = (hs, ppo)
res = pd.DataFrame(rows)
cols = KEYS + ["고정 대비 같은 변동성 수익 차", "고정 대비 CVaR5%(일)", "고정 대비 최대낙폭%"]

for code, nm in CODES.items():
    t = res[res["종목"] == code]
    base = t[t["seed"].isna()].set_index("정책")[cols]
    rl = t[t["seed"].notna()].groupby("정책", sort=False)[cols]
    mean, std = rl.mean(), rl.std()
    out = pd.concat([base, mean.rename(index=lambda x: f"{x} 시드 평균"), std.rename(index=lambda x: f"{x} 시드 표준편차")])
    print(f"=== {nm} ({len(ds[code][0])}일)")
    print(out.round(3).to_string())

print("\n[PPO 시드별 — 고정 대비 CVaR 차와 같은 변동성 수익 차]")
p = res[res["정책"] == "PPO"].pivot_table(index="seed", columns="종목",
                                         values=["평균 h", "고정 대비 CVaR5%(일)", "고정 대비 같은 변동성 수익 차"])
print(p.round(3).to_string())

print("\n[안전 범위·정수화 점검 (PPO, 시드 5개 합산)]")
rows2 = []
for code, (hs, ppo) in hs_all.items():
    allp = pd.concat(ppo.values())
    dh = pd.concat([p["h"].diff().abs() for p in ppo.values()])
    rows2.append({"종목": code, "하루 변화 상한에 걸린 날": f"{(abs(allp['want'] - allp['h']) > 0.0051).mean():.1%}",
                  "하루 평균 |Δh|": round(dh.mean(), 4), "h=0 비율": f"{(allp['h'] == 0).mean():.1%}",
                  "h=1 비율": f"{(allp['h'] >= 0.995).mean():.1%}",
                  "최대 증거금(명목 대비)": f"{allp['h'].max() * 0.36:.1%}"})
print(pd.DataFrame(rows2).to_string(index=False))

print("\n[연도별: 같은 평균 h 고정 대비 CVaR 차 (PPO 는 시드 평균)]")
yrows = []
for code, (hs, ppo) in hs_all.items():
    d, _ = ds[code]
    for y in YEARS:
        k = (d["date"].dt.year == y).values
        dy = d[k].reset_index(drop=True)
        fy = fixed_frontier(dy)
        for name, h in hs.items():
            if name.startswith("SAC"):
                continue
            hy = h[k].reset_index(drop=True)
            yrows.append({"종목": code, "연도": y, "정책": name.split(" s")[0] if name.startswith("PPO") else name,
                          "CVaR 차": vs_fixed(hedge_metrics(run_hedge(dy, hy)), fy)["CVaR5%(일)"]})
yr = pd.DataFrame(yrows).groupby(["종목", "연도", "정책"], sort=False)["CVaR 차"].mean().unstack()
print(yr.round(3).to_string())

print("\n[하루 지연 실행 — 고정 대비 CVaR 차 (PPO 는 시드 평균)]")
lrows = []
for code, (hs, _) in hs_all.items():
    d, _ = ds[code]
    for ex in ["close", "lag1"]:
        cfg = replace(HedgeConfig(), exec=ex)
        fr = fixed_frontier(d, cfg)
        for name, h in hs.items():
            if name.startswith("SAC"):
                continue
            lrows.append({"종목": code, "정책": name.split(" s")[0], "체결": ex,
                          "CVaR 차": vs_fixed(hedge_metrics(run_hedge(d, h, cfg)), fr)["CVaR5%(일)"]})
print(pd.DataFrame(lrows).pivot_table(index=["종목", "정책"], columns="체결", values="CVaR 차", sort=False).round(3)
      .to_string())

# ---- 분봉 강건성: 종가 단일가에 못 들어가면? (결정은 그날 종가로, 체결은 다음 날 09:05 분봉 종가) ------------------
print("\n[분봉 강건성: 선물 조정을 종가 단일가 대신 다음 날 09:05 분봉 종가에 체결했다면 (분봉이 있는 최근 약 1년)]")


def minute_px(code, day, fut_code, hhmm="09:05"):
    f = glob.glob(f"data/raw/kis/minutes/{day:%Y%m%d}/{code}_fut_{fut_code}.parquet")
    if not f:
        return np.nan
    m = pd.read_parquet(f[0])
    m = m[m["time"].dt.strftime("%H:%M") <= hhmm]
    return float(m["close"].iloc[-1]) if len(m) else np.nan


mrows = []
for code, (hs, _) in hs_all.items():
    d, _ = ds[code]
    first = pd.Timestamp(sorted(Path("data/raw/kis/minutes").iterdir())[0].name)
    k = (d["date"] >= first).values
    dm = d[k].reset_index(drop=True)
    nxt = dm["date"].shift(-1)
    # 같은 월물을 다음 날 아침에 체결한다. 다음 날이 없거나(마지막 날) 롤오버일은 제외
    px = pd.Series([minute_px(code, n, c) if pd.notna(n) else np.nan for n, c in zip(nxt, dm["front_code"])])
    gap_ticks = ((px - dm["front_settle"]).abs() / dm["front_settle"].map(tick_size)).median()
    for name, h in hs.items():
        if name.startswith("SAC"):
            continue
        r = run_hedge(dm, h[k].reset_index(drop=True))
        dn = r["contracts"].diff().fillna(r["contracts"]).where(~r["roll"], 0)
        # 매도 계약을 늘리면(dn>0) 체결가가 높을수록 이익, 줄이면 반대. 체결가 = 다음 날 09:05 vs 그날 정산가
        delta = (dn * 10 * (px - dm["front_settle"])).sum()
        yrs = len(dm) / 252
        mrows.append({"종목": code, "정책": name.split(" s")[0],
                      "체결가 변경 손익(bp/년)": delta / (1000 * dm["spot"].mean()) / yrs * 1e4,
                      "|체결가 차이| 합(bp/년)": (dn.abs() * 10 * (px - dm["front_settle"]).abs()).sum()
                      / (1000 * dm["spot"].mean()) / yrs * 1e4,
                      "연 거래 계약(롤오버 제외)": dn.abs().sum() / yrs})
    print(f"  {code}: {dm['date'].min():%Y-%m-%d}~{dm['date'].max():%Y-%m-%d} {len(dm)}일, "
          f"다음 날 09:05 분봉 종가와 그날 정산가 차이 중앙값 {gap_ticks:.1f}틱, 분봉 없는 날 {int(px.isna().sum())}일")
print(pd.DataFrame(mrows).groupby(["종목", "정책"], sort=False).mean().round(1).to_string())

plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=150, sharex=True)
for ax, (code, nm) in zip(axes, CODES.items()):
    hs, ppo = hs_all[code]
    d, _ = ds[code]
    P = pd.concat([p["h"].reset_index(drop=True) for p in ppo.values()], axis=1)
    ax.fill_between(d["date"], P.min(axis=1).rolling(5).mean(), P.max(axis=1).rolling(5).mean(), color="tab:green",
                    alpha=0.2, label="PPO 시드 범위")
    ax.plot(d["date"], P.mean(axis=1).rolling(5).mean(), color="tab:green", lw=1, label="PPO 시드 평균")
    ax.plot(d["date"], hs["규칙(변동성 목표)"].rolling(5).mean(), color="gray", lw=1, label="규칙(변동성 목표)")
    ax.plot(d["date"], hs["LGBM 변동성"].rolling(5).mean(), color="tab:blue", lw=0.8, label="LGBM 변동성")
    ax.set_ylabel(f"{nm} h (5일 평균)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="upper left", ncol=4)
fig.suptitle("PPO·LightGBM·규칙 기반 헤지 비율 (2022~)")
fig.tight_layout()
fig.savefig("outputs/ch09_h.png")
print("그림: outputs/ch09_h.png")
