"""헤지 의사결정용 피처·레이블 패널 (8장).

피처는 모두 그날(t) 종가까지의 정보로 만든다. 레이블은 t+1 ~ t+H 거래일의 미래 값이다.
학습에 쓸 수 있는 행은 '레이블이 끝나는 날(label_end)'이 학습 기간 안에 있는 행뿐이다(8-2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

H = 5  # 예측 기간(거래일)

FEATURES = [
    "ret_1", "ret_5", "ret_20", "ret_60", "vol_5", "vol_20", "vol_60", "vol_ratio", "dd_60",
    "ma20_gap", "ma60_gap", "vwap5_gap", "volume_ratio", "basis_pct", "mis_pct", "mis_z", "dte",
    "mkt_ret_5", "mkt_vol_20", "rel_ret_20",
]
# rate(금리 수준)는 패널에 남기되 피처에서 뺀다: 수준 변수는 사실상 '시점 표지'라서
# 학습 범위를 벗어난 연도(2022 금리 급등)에 나무 모델이 외삽하지 못한다(8-2 범위 이탈 점검).
TARGETS = ["y_crash", "y_up", "y_logvol"]


def stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """한 종목의 일별 피처(그날까지의 정보만)와 레이블(미래)."""
    f = df[["date", "spot", "spot_volume", "spot_value", "basis_pct", "mis_pct", "mis_z", "dte", "rate"]].copy()
    lr = np.log(f["spot"]).diff()
    f["lr"] = lr
    for k in (1, 5, 20, 60):
        f[f"ret_{k}"] = f["spot"].pct_change(k)
    for k in (5, 20, 60):
        f[f"vol_{k}"] = lr.rolling(k, min_periods=k).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_5"] / f["vol_20"]
    f["dd_60"] = f["spot"] / f["spot"].rolling(60, min_periods=20).max() - 1
    f["ma20_gap"] = f["spot"] / f["spot"].rolling(20).mean() - 1
    f["ma60_gap"] = f["spot"] / f["spot"].rolling(60).mean() - 1
    vwap5 = f["spot_value"].rolling(5).sum() / f["spot_volume"].rolling(5).sum()
    f["vwap5_gap"] = f["spot"] / vwap5 - 1
    f["volume_ratio"] = f["spot_volume"] / f["spot_volume"].rolling(20).mean()

    # 레이블: t+1 ~ t+H
    fwd = np.log(f["spot"].shift(-H) / f["spot"])
    fwd_vol = lr[::-1].rolling(H, min_periods=H).std()[::-1].shift(-1) * np.sqrt(252)  # t+1..t+H 의 표준편차
    thr = -f["vol_20"] * np.sqrt(H / 252)  # 그날 변동성 기준 −1σ (임계값도 t 시점 정보)
    f["fwd_ret"] = fwd
    f["y_crash"] = np.where(fwd.notna() & thr.notna(), (fwd < thr).astype(float), np.nan)  # 임계값 없으면 비움
    f["y_up"] = np.where(fwd.notna(), (fwd > 0).astype(float), np.nan)
    f["y_logvol"] = np.log(fwd_vol)
    f["fwd_vol"] = fwd_vol
    f["label_end"] = f["date"].shift(-H)
    return f


def build_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """여러 종목을 쌓고, 같은 날 종목들의 평균으로 시장 피처를 만든다(그날까지의 정보)."""
    parts = [stock_features(df).assign(code=code) for code, df in frames.items()]
    p = pd.concat(parts, ignore_index=True)
    mkt = p.groupby("date").agg(mkt_lr=("lr", "mean")).sort_index()
    mkt["mkt_ret_5"] = mkt["mkt_lr"].rolling(5).sum()
    mkt["mkt_vol_20"] = mkt["mkt_lr"].rolling(20, min_periods=20).std() * np.sqrt(252)
    mkt["mkt_ret_20"] = mkt["mkt_lr"].rolling(20).sum()
    p = p.merge(mkt[["mkt_ret_5", "mkt_vol_20", "mkt_ret_20"]], left_on="date", right_index=True, how="left")
    p["rel_ret_20"] = np.log1p(p["ret_20"]) - p["mkt_ret_20"]
    return p.sort_values(["date", "code"]).reset_index(drop=True)
