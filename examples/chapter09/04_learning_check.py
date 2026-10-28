"""09-3. 학습이 됐는가: 학습 곡선과 학습 구간 안(in-sample) 점검.

1) 에피소드 보상: 학습 초반 10% 에피소드 평균 vs 마지막 10% 평균 (시험 연도·시드별)
2) 학습 구간 안에서 결정론적 정책의 일평균 보상 vs 같은 구간의 '보상 기준 최적 고정 h'
   — 학습 데이터에서조차 고정 h 를 못 이기면 시험 구간에서 기대할 것이 없다.
3) 같은 비교를 시험 연도에서 (고정 h 는 학습 구간에서 고른 값) — 학습·시험 격차 = 과적합의 크기
선행: 03_train.py
실행: uv run python examples/chapter09/04_learning_check.py [--algo sac]
"""

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from hedging import plotting
from hedging.rl.env import EnvConfig, HedgeEnv
from hedging.rl.train import load_frames, train_frames

ap = argparse.ArgumentParser()
ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
algo = ap.parse_args().algo
Model = PPO if algo == "ppo" else SAC
pd.set_option("display.width", 200)
YEARS = [2022, 2023, 2024, 2025, 2026]
SEEDS = [1, 2, 3, 4, 5]
D = "outputs/ch09_models"
panel = pd.read_parquet("data/processed/ch08_panel.parquet")
frames = load_frames(panel)


def mean_reward(f, stats, act):
    """한 종목 학습 구간 전체를 지난 일평균 보상. act(obs) → 행동."""
    env = HedgeEnv([f], stats, train=False, start=0)
    obs, _ = env.reset()
    rs, done = [], False
    while not done:
        obs, r, _, done, _ = env.step(act(obs))
        rs.append(r)
    return np.mean(rs)


def test_env(f, stats, y, cfg=EnvConfig()):
    idx = np.flatnonzero(f["date"].dt.year.values == y)
    return HedgeEnv([f], stats, cfg=cfg, train=False, start=int(idx[0]), end=int(min(idx[-1] + 1, len(f) - 1)))


def test_reward(f, stats, y, act, cfg=EnvConfig()):
    env = test_env(f, stats, y, cfg)
    obs, _ = env.reset()
    rs, done = [], False
    while not done:
        obs, r, _, done, _ = env.step(act(obs))
        rs.append(r)
    return np.mean(rs)


def fixed_reward(f, stats, h):
    env = HedgeEnv([f], stats, cfg=replace(EnvConfig(), max_step=1.0), train=False, start=0)
    env.reset()
    a = np.array([2 * h - 1], dtype=np.float32)
    rs, done = [], False
    while not done:
        _, r, _, done, _ = env.step(a)
        rs.append(r)
    return np.mean(rs)


rows, curves = [], {}
for y in YEARS:
    tr = train_frames(frames, y)
    grid = np.round(np.arange(0, 1.01, 0.1), 1)
    for s in SEEDS:
        path = f"{D}/{algo}_{y}_s{s}"
        mon = pd.read_csv(path + "_monitor.monitor.csv", skiprows=1)
        n = max(1, len(mon) // 10)
        curves[(y, s)] = mon["r"].rolling(50, min_periods=10).mean().values
        model = Model.load(path, device="cpu")
        stats = pd.read_pickle(path + "_stats.pkl")
        pol = np.mean([mean_reward(f, stats, lambda o: model.predict(o, deterministic=True)[0]) for f in tr])
        fx = {h: np.mean([fixed_reward(f, stats, h) for f in tr]) for h in grid}
        best = max(fx, key=fx.get)
        te = list(frames.values())
        pol_te = np.mean([test_reward(f, stats, y, lambda o: model.predict(o, deterministic=True)[0]) for f in te])
        a_best = np.array([2 * best - 1], dtype=np.float32)
        fx_te = np.mean([test_reward(f, stats, y, lambda o: a_best, replace(EnvConfig(), max_step=1.0)) for f in te])
        rows.append({"시험 연도": y, "seed": s, "에피소드 수": len(mon), "초반 10% 보상": mon["r"].head(n).mean(),
                     "마지막 10% 보상": mon["r"].tail(n).mean(), "정책 일평균 보상": pol,
                     "최적 고정 h": best, "최적 고정 일평균 보상": fx[best], "정책 − 최적 고정": pol - fx[best],
                     "시험: 정책 보상": pol_te, "시험: 고정 보상": fx_te, "시험: 정책 − 고정": pol_te - fx_te})
t = pd.DataFrame(rows)
print(f"[{algo.upper()} 학습 점검 — 보상 단위: 명목금액 대비 %]")
print(t.groupby("시험 연도")[["에피소드 수", "초반 10% 보상", "마지막 10% 보상", "정책 일평균 보상", "최적 고정 h",
                           "최적 고정 일평균 보상", "정책 − 최적 고정"]].mean().round(4).to_string())
print(f"\n학습 구간에서 최적 고정 h 보다 보상이 높은 모델: {(t['정책 − 최적 고정'] > 0).sum()} / {len(t)}")
print(t.pivot_table(index="시험 연도", columns="seed", values="정책 − 최적 고정").round(4).to_string())
print("\n[같은 모델을 시험 연도에서: 정책 vs 학습 구간에서 고른 고정 h — 일평균 보상]")
print(t.groupby("시험 연도")[["정책 일평균 보상", "시험: 정책 보상", "시험: 고정 보상", "시험: 정책 − 고정"]].mean()
      .round(4).to_string())
print(f"시험 연도에서 고정 h 보다 보상이 높은 모델: {(t['시험: 정책 − 고정'] > 0).sum()} / {len(t)}")

plotting.setup()
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
for y, c in zip(YEARS, plt.cm.viridis(np.linspace(0, 0.9, len(YEARS)))):
    L = min(len(curves[(y, s)]) for s in SEEDS)
    m = np.nanmean([curves[(y, s)][:L] for s in SEEDS], axis=0)
    ax.plot(np.arange(L), m, color=c, lw=1, label=f"시험 {y}용 (학습 ~{y - 1})")
ax.set_xlabel("에피소드 (126거래일)")
ax.set_ylabel("에피소드 보상 (50개 이동평균)")
ax.set_title(f"{algo.upper()} 학습 곡선 (시드 5개 평균)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"outputs/ch09_curve_{algo}.png")
print(f"그림: outputs/ch09_curve_{algo}.png")
