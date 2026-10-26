"""헤지 비율 정책 (7장). 모두 그날 종가까지의 정보만 써서 h ∈ [0, 1]을 정한다.

- fixed:      항상 같은 h
- ma_trend:   현물이 이동평균 아래면 많이(h_down), 위면 적게(h_up) 헤지
- vwap:       현물 종가가 최근 N일 VWAP(거래대금 ÷ 거래량) 아래면 많이 헤지 — 매도 압력 신호
- vol_target: 변동성이 기준보다 높을 때만 헤지해 남는 노출의 변동성을 기준 근처로 — h = 1 − σ* / σ
- combo:      ma_trend 와 vol_target 의 평균

매개변수는 미리 정한 값이며 데이터로 최적화하지 않는다(7-4에서 같은 평균 h의 고정 헤지와 비교).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixed(df: pd.DataFrame, h: float) -> pd.Series:
    return pd.Series(h, index=df.index)


def ma_trend(df: pd.DataFrame, window: int = 60, h_down: float = 0.8, h_up: float = 0.2) -> pd.Series:
    ma = df["spot"].rolling(window, min_periods=window).mean()
    h = np.where(df["spot"] < ma, h_down, h_up)
    return pd.Series(np.where(ma.isna(), 0.5, h), index=df.index)


def vwap(df: pd.DataFrame, window: int = 5, h_below: float = 0.8, h_above: float = 0.2) -> pd.Series:
    v = df["spot_value"].rolling(window, min_periods=window).sum() / df["spot_volume"].rolling(
        window, min_periods=window).sum()
    h = np.where(df["spot"] < v, h_below, h_above)
    return pd.Series(np.where(v.isna(), 0.5, h), index=df.index)


def realized_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return np.log(df["spot"]).diff().rolling(window, min_periods=window).std() * np.sqrt(252)


def vol_target(df: pd.DataFrame, window: int = 20, min_history: int = 120) -> pd.Series:
    """σ* = 그날까지 관측된 20일 변동성의 중앙값(확장 창). 변동성이 그보다 높을수록 많이 헤지한다."""
    vol = realized_vol(df, window)
    target = vol.expanding(min_history).median()
    h = (1 - target / vol).clip(0, 1)
    return h.fillna(0.5)


def combo(df: pd.DataFrame) -> pd.Series:
    return (ma_trend(df) + vol_target(df)) / 2


POLICIES = {
    "추세(60일 이동평균)": ma_trend,
    "VWAP(5일)": vwap,
    "변동성 목표": vol_target,
    "추세+변동성": combo,
}
