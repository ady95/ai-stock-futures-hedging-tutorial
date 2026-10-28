import numpy as np
import pandas as pd
import pytest

from hedging.backtest.hedge_ledger import run_hedge
from hedging.rl.env import OBS_COLS, EnvConfig, HedgeEnv, RewardConfig, prepare_frame


def _frame(n=80, ltd_idx=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n)
    spot = 50_000 + np.cumsum(rng.normal(0, 500, n)).round(-1)
    fut = spot + 200
    df = pd.DataFrame({"date": dates, "spot": spot, "front_code": "A", "front_close": fut, "front_settle": fut,
                       "next_code": "B", "next_close": fut + 300, "ltd": dates[ltd_idx], "rate": 0.03})
    after = df.index > ltd_idx  # 최종거래일 다음 날부터 차월물
    for c in ["front_close", "front_settle"]:
        df.loc[after, c] = df.loc[after, "spot"] + 300
    df.loc[after, "ltd"] = dates[-1] + pd.Timedelta(days=30)
    feats = pd.DataFrame({"date": dates, **{c: rng.normal(size=n) for c in OBS_COLS if c != "vol_rel"}})
    feats["vol_20"] = np.abs(feats["vol_20"]) + 0.1
    return prepare_frame(df, feats)


def _stats():
    return pd.Series(0.0, index=OBS_COLS), pd.Series(1.0, index=OBS_COLS)


def test_env_pnl_matches_ledger_including_rollover():
    f = _frame()
    env = HedgeEnv([f], _stats(), cfg=EnvConfig(max_step=1.0), train=False, start=1)
    env.reset()
    rng = np.random.default_rng(1)
    hs, won, done = [], 0.0, False
    while not done:
        base = 1000 * f["spot"].iat[env.i]
        _, _, _, done, info = env.step(rng.uniform(-1, 1, 1).astype(np.float32))
        hs.append(info["h"])
        won += (info["dv"] - info["cost"]) * base / 100
    d = f.iloc[1:1 + len(hs) + 1].reset_index(drop=True)
    res = run_hedge(d, pd.Series(hs + [hs[-1]]))
    assert won == pytest.approx(res["pnl"].sum(), rel=1e-9)


def test_action_mapping_safety_and_integer_contracts():
    f = _frame()
    env = HedgeEnv([f], _stats(), train=False, start=5, init_h=0.3)
    env.reset()
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["want"] == 1.0 and info["h"] == pytest.approx(0.5)  # 하루 +0.2 상한
    _, _, _, _, info = env.step(np.array([-0.333], dtype=np.float32))
    assert info["h"] == pytest.approx(0.33)  # 0.3335 → 33계약
    assert round(info["h"] * 100) == info["h"] * 100


def test_cost_counted_once_in_reward():
    f = _frame()
    rc = RewardConfig(lam_var=0, lam_dd=0, lam_turn=0, margin_rate=0)
    env = HedgeEnv([f], _stats(), rc, EnvConfig(max_step=1.0), train=False, start=3)
    env.reset()
    _, r, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert r == pytest.approx(info["dv"] - info["cost"])


def test_train_episodes_have_fixed_length():
    f = _frame(200)
    env = HedgeEnv([f], _stats(), cfg=EnvConfig(episode_len=20), train=True)
    env.reset(seed=0)
    steps, done = 0, False
    while not done:
        _, _, _, done, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 20
