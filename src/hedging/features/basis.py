"""괴리(mispricing) 통계 피처 (4장).

모든 피처는 그날까지의 정보만 쓴다(rolling 창에 미래 값이 들어가지 않는다).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(fv: pd.DataFrame, window: int = 20, big_move_pct: float = 5.0) -> pd.DataFrame:
    s = fv.copy()
    roll = s["mis_pct"].rolling(window, min_periods=window)
    s["mis_z"] = (s["mis_pct"] - roll.mean()) / roll.std()
    s["spot_ret_pct"] = s["spot"].pct_change() * 100
    s["big_move"] = s["spot_ret_pct"].abs() >= big_move_pct  # 종가 시각 차이로 괴리가 왜곡되기 쉬운 날
    s["near_expiry"] = s["dte"] <= 2
    s["thin"] = s["front_volume"] < s["front_volume"].rolling(60, min_periods=20).median() * 0.2
    s["usable"] = ~(s["big_move"] | s["near_expiry"] | s["thin"] | s["roll"])
    return s


def half_life(mis: pd.Series, contract: pd.Series) -> tuple[float, float]:
    """같은 월물 안에서 x_t = φ·x_{t-1} + ε 를 추정한 φ와 반감기(거래일)."""
    prev = mis.groupby(contract).shift(1)
    ok = prev.notna() & mis.notna()
    x, y = prev[ok].values, mis[ok].values
    phi = float((x @ y) / (x @ x))
    hl = float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("inf")
    return phi, hl
