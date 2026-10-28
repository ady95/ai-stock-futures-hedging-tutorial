"""헤지 강화학습 환경 (9장). Gymnasium 인터페이스.

한 스텝 = 하루. 그날 종가까지의 관측을 보고 행동을 고르면, 그날 선물 종가에 계약 수를 맞추고(비용 발생),
다음 날까지의 평가손익을 보상으로 받는다. 손익 계산은 7장 run_hedge 와 같다(테스트로 확인).

행동  a ∈ [-1, 1] → 원하는 헤지 비율 (a + 1) / 2 → 안전 범위: 하루 변화 ±max_step, 0 ≤ h ≤ 1
       → 계약 정수화 N = round(qty × h / 승수). 정수화·범위 제한은 모두 환경 안에서 한다.
관측  시장 피처(학습 구간 평균·표준편차로 표준화) + 현재 h + 헤지 포트폴리오의 최근 60일 낙폭
보상  r = ΔV − 비용 − λ_var·ΔV² − λ_dd·낙폭 심화 − λ_turn·|Δh| − 증거금 기회비용   (모두 명목금액 대비 %)
       ΔV 는 비용 차감 전 평가손익이다. 비용은 별도 항으로 한 번만 뺀다(이중 차감 금지).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd

from hedging.instruments.contracts import tick_size

OBS_COLS = ["vol_5", "vol_20", "vol_60", "vol_ratio", "vol_rel", "ret_5", "ret_20", "ret_60", "dd_60",
            "ma20_gap", "ma60_gap", "mkt_ret_5", "mkt_vol_20", "basis_pct", "dte"]


@dataclass(frozen=True)
class RewardConfig:
    lam_var: float = 0.02  # 위험 회피 계수: ΔV(%)² 에 곱한다
    lam_dd: float = 0.1  # 최근 60일 낙폭이 깊어진 만큼(%p)
    lam_turn: float = 0.01  # |Δh| 당 (%)
    margin_rate: float = 0.36  # 증거금률(2-2)
    funding_spread: float = 0.0  # 증거금 기회비용 = 증거금 × (금리 + spread) / 252


@dataclass(frozen=True)
class EnvConfig:
    qty: int = 1_000
    multiplier: int = 10
    fut_fee: float = 0.00003
    slip_ticks: float = 0.5
    max_step: float = 0.2  # 하루 h 변화 상한 (안전 범위)
    episode_len: int = 126  # 학습 에피소드 길이(거래일)
    dd_window: int = 60


def prepare_frame(df: pd.DataFrame, panel_code: pd.DataFrame) -> pd.DataFrame:
    """load_hedge_frame 결과 + 8장 패널 피처 → 환경용 표. 손익 계산에 필요한 가격 열을 미리 만든다."""
    d = df.merge(panel_code.drop(columns=[c for c in panel_code.columns if c in df.columns and c != "date"]),
                 on="date", how="left").sort_values("date").reset_index(drop=True)
    d["vol_rel"] = d["vol_20"] / d["vol_20"].expanding(120).median()  # 7장 규칙의 입력 σ/σ*
    is_ltd = (d["date"] >= d["ltd"]).values
    settle = d["front_settle"].fillna(d["front_close"]).values
    d["is_ltd"] = is_ltd
    d["p_now"] = np.where(is_ltd, d["spot"], settle)  # 그날 평가·체결 가격(최종거래일은 최종결제 = 현물 종가)
    d["p_next"] = d["next_close"].fillna(d["spot"]).values  # 최종거래일 롤오버 체결가
    ref = np.where(is_ltd, d["p_next"], d["p_now"])  # 다음 날 손익의 기준 가격
    d["fut_move"] = d["p_now"] - pd.Series(ref).shift(1).values  # 1계약(승수 제외) 하루 가격 변화
    d["spot_move"] = d["spot"].diff()
    return d


def trade_cost(n_old: int, n_new: int, is_ltd: bool, p_now: float, p_next: float, cfg: EnvConfig) -> float:
    """그날 n_old → n_new 로 맞출 때의 비용(원). run_hedge 와 같은 규칙."""
    m = cfg.multiplier
    if is_ltd:  # 만기 월물은 최종결제(슬리피지 없음), 차월물을 새로 매도
        c = n_old * m * cfg.fut_fee * p_now
        if n_new > 0:
            c += n_new * m * (cfg.fut_fee * p_next + cfg.slip_ticks * tick_size(p_next))
        return c
    dn = abs(n_new - n_old)
    return dn * m * (cfg.fut_fee * p_now + cfg.slip_ticks * tick_size(p_now)) if dn else 0.0


class HedgeEnv(gym.Env):
    """frames: prepare_frame 결과 목록(종목별). stats: OBS_COLS 의 (평균, 표준편차) — 학습 구간에서만 계산.
    train=True 면 에피소드마다 종목·시작일을 무작위로 고르고, False 면 frames[0] 의 [start, end) 를 한 번 지난다."""

    metadata = {"render_modes": []}

    def __init__(self, frames, stats, rcfg=RewardConfig(), cfg=EnvConfig(), train=True, start=0, end=None,
                 init_h=0.0):
        super().__init__()
        self.frames = [f.reset_index(drop=True) for f in frames]
        self.mu, self.sd = stats
        self.rcfg, self.cfg, self.train = rcfg, cfg, train
        self.start, self.end, self.init_h = start, end, init_h
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(len(OBS_COLS) + 2,), dtype=np.float32)
        self._X = [np.clip(((f[OBS_COLS] - self.mu) / self.sd).fillna(0).values, -5, 5).astype(np.float32)
                   for f in self.frames]
        cols = ["spot", "spot_move", "fut_move", "p_now", "p_next", "rate"]
        self._A = [{c: f[c].fillna(0).to_numpy(float) for c in cols} | {"is_ltd": f["is_ltd"].to_numpy(bool),
                                                                       "date": f["date"].to_numpy()}
                   for f in self.frames]

    # ---- 상태
    def _obs(self):
        dd = self.values[-1] - max(self.values[-self.cfg.dd_window:])
        return np.concatenate([self._X[self.k][self.i], [self.h * 2 - 1, dd / 10]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.train:
            self.k = int(self.np_random.integers(len(self.frames)))
            n = len(self.frames[self.k])
            self.i = int(self.np_random.integers(0, n - self.cfg.episode_len - 1))
            self.stop = self.i + self.cfg.episode_len
        else:
            self.k, self.i = 0, self.start
            self.stop = (self.end if self.end is not None else len(self.frames[0])) - 1
        self.h = float(self.init_h)
        self.n = int(np.floor(self.cfg.qty * self.h / self.cfg.multiplier + 0.5))
        self.values = [0.0]
        return self._obs(), {}

    def step(self, action):
        cfg, rc = self.cfg, self.rcfg
        a, i = self._A[self.k], self.i
        want = (float(np.clip(action[0], -1, 1)) + 1) / 2
        h_new = float(np.clip(want, self.h - cfg.max_step, self.h + cfg.max_step))
        n_new = int(np.floor(cfg.qty * h_new / cfg.multiplier + 0.5))  # 계약 정수화
        base = cfg.qty * a["spot"][i]
        cost = trade_cost(self.n, n_new, a["is_ltd"][i], a["p_now"][i], a["p_next"][i], cfg) / base * 100
        dh = abs(n_new - self.n) * cfg.multiplier / cfg.qty
        self.n, self.h = n_new, n_new * cfg.multiplier / cfg.qty
        # 다음 날까지의 평가손익(비용 차감 전)
        dv = (cfg.qty * a["spot_move"][i + 1] - self.n * cfg.multiplier * a["fut_move"][i + 1]) / base * 100
        margin = self.n * cfg.multiplier * a["p_now"][i] * rc.margin_rate / base * 100
        funding = margin * (a["rate"][i] + rc.funding_spread) / 252
        self.values.append(self.values[-1] + dv - cost)
        prev_dd = self.values[-2] - max(self.values[-cfg.dd_window - 1:-1])
        new_dd = self.values[-1] - max(self.values[-cfg.dd_window:])
        deeper = max(0.0, prev_dd - new_dd)
        reward = dv - cost - rc.lam_var * dv ** 2 - rc.lam_dd * deeper - rc.lam_turn * dh - funding
        self.i += 1
        done = self.i >= self.stop
        info = {"date": a["date"][i], "h": self.h, "want": want, "dv": dv, "cost": cost, "margin": margin}
        return self._obs(), float(reward), False, bool(done), info


def obs_stats(frames, cutoff: pd.Timestamp):
    """관측 표준화 통계: cutoff 이전 행만 쓴다(시험 구간 정보 금지)."""
    x = pd.concat([f.loc[f["date"] < cutoff, OBS_COLS] for f in frames])
    return x.mean(), x.std().replace(0, 1)
