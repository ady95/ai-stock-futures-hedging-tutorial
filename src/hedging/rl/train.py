"""PPO(·SAC) 워크포워드 학습과 평가 (9장).

시험 연도 y 마다: y년 1월 1일 − 8일(엠바고) 이전 데이터로만 학습하고, y년을 결정론적 정책으로 지난다.
하이퍼파라미터·보상 계수·학습 스텝 수는 미리 정한 값이며 시험 결과로 고르지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from hedging.backtest.hedge_ledger import load_hedge_frame
from hedging.rl.env import OBS_COLS, EnvConfig, HedgeEnv, RewardConfig, obs_stats, prepare_frame

PPO_PARAMS = dict(learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, gamma=0.95, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64]))
SAC_PARAMS = dict(learning_rate=3e-4, buffer_size=200_000, batch_size=256, gamma=0.95, tau=0.005,
                  learning_starts=5_000, policy_kwargs=dict(net_arch=[64, 64]))
TIMESTEPS = 200_000
CODES = ["005930", "000660"]


def cutoff_of(year: int, embargo_days: int = 5) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=int(embargo_days * 7 / 5) + 1)


def load_frames(panel: pd.DataFrame, codes=CODES) -> dict[str, pd.DataFrame]:
    return {c: prepare_frame(load_hedge_frame(c), panel[panel["code"] == c]) for c in codes}


def train_frames(frames: dict[str, pd.DataFrame], year: int) -> list[pd.DataFrame]:
    """학습용: cutoff 이전, 20일 변동성이 생긴 날부터. 아직 계산되지 않은 긴 창 피처(60일 수익률, σ/σ* 등)는
    표준화 후 0(학습 평균)으로 채운다 — 2020년 3월 급락을 학습에서 빼지 않기 위해서다.
    마지막 행은 다음 날 손익이 없으므로 환경이 행동하지 않는다."""
    c = cutoff_of(year)
    return [f[(f["date"] < c) & f["vol_20"].notna() & f["fut_move"].notna()].reset_index(drop=True)
            for f in frames.values()]


def train_one(frames, year: int, seed: int, algo: str = "ppo", timesteps: int = TIMESTEPS,
              rcfg=RewardConfig(), cfg=EnvConfig(), out_dir: str = "outputs/ch09_models") -> str:
    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    tr = train_frames(frames, year)
    stats = obs_stats(tr, cutoff_of(year))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"{algo}_{year}_s{seed}")
    env = Monitor(HedgeEnv(tr, stats, rcfg, cfg, train=True), path + "_monitor")  # 에피소드 보상 기록(학습 곡선)
    env.reset(seed=seed)
    Model, params = (PPO, PPO_PARAMS) if algo == "ppo" else (SAC, SAC_PARAMS)
    model = Model("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **params)
    model.learn(total_timesteps=timesteps)
    model.save(path)
    pd.to_pickle(stats, path + "_stats.pkl")
    return path


def run_policy(frames: dict[str, pd.DataFrame], code: str, years: list[int], paths: dict[int, str],
               algo: str = "ppo", rcfg=RewardConfig(), cfg=EnvConfig()) -> pd.DataFrame:
    """시험 연도들을 이어서 지난다. 연도마다 그 해의 모델·표준화 통계를 쓰고, 보유 h·낙폭 상태는 이어받는다.
    반환: date, h (그날 종가에 맞춘 헤지 비율)."""
    from stable_baselines3 import PPO, SAC

    Model = PPO if algo == "ppo" else SAC
    f = frames[code]
    rows, h, values = [], 0.0, [0.0]
    for y in years:
        model = Model.load(paths[y], device="cpu")
        stats = pd.read_pickle(paths[y] + "_stats.pkl")
        idx = np.flatnonzero(f["date"].dt.year.values == y)
        end = idx[-1] + 1 if y != years[-1] else idx[-1]  # 마지막 해는 다음 날 손익이 없으므로 하루 전까지
        env = HedgeEnv([f], stats, rcfg, cfg, train=False, start=int(idx[0]), end=int(end) + 1, init_h=h)
        obs, _ = env.reset()
        env.values = list(values[-cfg.dd_window:])
        obs = env._obs()
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, _, done, info = env.step(act)
            rows.append({"date": info["date"], "h": info["h"], "want": info["want"]})
        h, values = env.h, env.values
    return pd.DataFrame(rows)
