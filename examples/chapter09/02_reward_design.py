"""09-2. 보상 설계 점검: 고정 h 정책을 환경에 넣어 보상의 각 항을 분해한다.

학습 전에, 이 보상이 어떤 h 를 좋아하는지(보상 기준 최적 고정 h)와 항별 크기를 본다.
기간은 첫 학습 구간(2020~2021, 시험 2022의 학습 데이터)만 쓴다 — 시험 구간을 보지 않는다.
실행: uv run python examples/chapter09/02_reward_design.py
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from hedging.rl.env import EnvConfig, HedgeEnv, RewardConfig, obs_stats
from hedging.rl.train import cutoff_of, load_frames, train_frames

pd.set_option("display.width", 200)
panel = pd.read_parquet("data/processed/ch08_panel.parquet")
frames = load_frames(panel)
tr = train_frames(frames, 2022)
stats = obs_stats(tr, cutoff_of(2022))
print("학습 구간:", ", ".join(f"{c} {f['date'].min():%Y-%m-%d}~{f['date'].max():%Y-%m-%d} ({len(f)}일)"
                            for c, f in zip(frames, tr)))


def rollout(f, h, rcfg=RewardConfig(), cfg=EnvConfig()):
    """고정 h 정책으로 한 종목 학습 구간 전체를 지나며 항별 합계를 낸다(일평균, %)."""
    env = HedgeEnv([f], stats, rcfg, replace(cfg, max_step=1.0), train=False, start=0)
    env.reset()
    a = np.array([2 * h - 1], dtype=np.float32)
    rows, done = [], False
    while not done:
        prev_vals = list(env.values)
        _, r, _, done, info = env.step(a)
        v = env.values
        prev_dd = prev_vals[-1] - max(prev_vals[-cfg.dd_window:])
        new_dd = v[-1] - max(v[-cfg.dd_window:])
        dh = info["h"] - (rows[-1]["h"] if rows else 0.0)
        rows.append({"h": info["h"], "dv": info["dv"], "cost": info["cost"], "reward": r,
                     "var": rcfg.lam_var * info["dv"] ** 2, "dd": rcfg.lam_dd * max(0.0, prev_dd - new_dd),
                     "turn": rcfg.lam_turn * abs(dh), "funding": info["margin"] * 0.0})
    t = pd.DataFrame(rows)
    t["funding"] = t["reward"] - (t["dv"] - t["cost"] - t["var"] - t["dd"] - t["turn"])
    t["funding"] = -t["funding"]
    return t


rows = []
for h in np.round(np.arange(0, 1.01, 0.1), 1):
    parts = pd.concat([rollout(f, h) for f in tr])
    m = parts.mean()
    rows.append({"고정 h": h, "ΔV(비용 전)": m["dv"], "비용": -m["cost"], "분산 페널티": -m["var"],
                 "낙폭 페널티": -m["dd"], "회전 페널티": -m["turn"], "증거금 기회비용": -m["funding"],
                 "보상 합": m["reward"]})
t = pd.DataFrame(rows).set_index("고정 h")
print("\n[고정 h 별 보상 항 — 일평균, 명목금액 대비 %, 두 종목 학습 구간]")
print((t * 1).round(4).to_string())
print(f"보상 기준 최적 고정 h: {t['보상 합'].idxmax()}")

print("\n[위험 회피 계수 λ_var 에 따른 보상 기준 최적 고정 h (참고용, 본문 실험은 0.02 고정)]")
out = []
for lam in [0.0, 0.01, 0.02, 0.05, 0.1]:
    rc = replace(RewardConfig(), lam_var=lam)
    s = {h: pd.concat([rollout(f, h, rc) for f in tr])["reward"].mean() for h in np.round(np.arange(0, 1.01, 0.1), 1)}
    out.append({"λ_var": lam, "최적 고정 h": max(s, key=s.get)})
print(pd.DataFrame(out).to_string(index=False))

print("\n[비용 이중 차감 점검: 하루 보상 = ΔV − 비용 − 페널티, 장부 손익 = ΔV − 비용]")
p = rollout(tr[0], 0.5)
gap = (p["dv"] - p["cost"]).sum() - (p["reward"] + p["var"] + p["dd"] + p["turn"] + p["funding"]).sum()
print(f"  (ΔV − 비용) 합과 (보상 + 페널티) 합의 차이: {gap:.2e}  → 비용은 보상에 한 번만 들어간다")
