"""09-1. 헤지 환경 점검: 관측·행동 공간, 행동 매핑, 7장 장부와의 손익 일치.

실행: uv run python examples/chapter09/01_env_check.py
"""

from dataclasses import replace

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from hedging.backtest.hedge_ledger import run_hedge
from hedging.rl.env import OBS_COLS, EnvConfig, HedgeEnv, obs_stats
from hedging.rl.train import cutoff_of, load_frames, train_frames

pd.set_option("display.width", 200)
panel = pd.read_parquet("data/processed/ch08_panel.parquet")
frames = load_frames(panel)
tr = train_frames(frames, 2022)
stats = obs_stats(tr, cutoff_of(2022))

env = HedgeEnv(tr, stats)
check_env(env, skip_render_check=True)  # Gymnasium 규약 검사(통과하면 조용하다)
print("관측 공간:", env.observation_space.shape, "= 피처", len(OBS_COLS), "+ 현재 h + 최근 60일 낙폭")
print("행동 공간:", env.action_space)

print("\n[행동 매핑: 현재 h = 0.30, 하루 변화 상한 0.2]")
rows = []
for a in [-1.0, -0.5, 0.0, 0.3, 0.6, 1.0]:
    e = HedgeEnv([tr[0]], stats, train=False, start=100, init_h=0.3)
    e.reset()
    _, _, _, _, info = e.step(np.array([a], dtype=np.float32))
    rows.append({"행동 a": a, "원하는 h": round(info["want"], 3), "적용 h": info["h"],
                 "계약 수": int(round(info["h"] * 100))})
print(pd.DataFrame(rows).to_string(index=False))

print("\n[7장 장부와의 일치: 삼성전자 2022-01 ~ 끝, 무작위 행동]")
f = frames["005930"]
start = int((f["date"] >= "2022-01-01").idxmax())
e = HedgeEnv([f], stats, cfg=replace(EnvConfig(), max_step=1.0), train=False, start=start)
e.reset()
rng = np.random.default_rng(0)
hs, won, done = [], 0.0, False
while not done:
    base = 1000 * f["spot"].iat[e.i]
    _, _, _, done, info = e.step(rng.uniform(-1, 1, 1).astype(np.float32))
    hs.append(info["h"])
    won += (info["dv"] - info["cost"]) * base / 100
d = f.iloc[start:start + len(hs) + 1].reset_index(drop=True)
res = run_hedge(d, pd.Series(hs + [hs[-1]]))
print(f"  환경 손익 합 {won:,.0f}원, 7장 run_hedge 손익 합 {res['pnl'].sum():,.0f}원, {len(hs)}스텝")

print("\n[관측 예시: 2022-01-03 삼성전자 (표준화 값)]")
x = pd.Series(e._X[0][start], index=OBS_COLS).round(2)
print(x.to_frame("z").T.to_string(index=False))
