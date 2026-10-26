"""예측 기반 헤지 비율 (8장). 8-3의 워크포워드 예측을 종목 시계열에 붙여 h 를 만든다.

- 규칙(7장)      : 변동성 목표, h = 1 − σ*/σ(지난 20일)
- HAR           : 선형 변동성 예측 σ̂ 로 같은 식 — 'LightGBM이 아니어도 되는가'의 기준선
- LGBM 변동성    : LightGBM 변동성 예측 σ̂ 로 같은 식
- LGBM 상단80%   : 분위 회귀 예측(보수적 σ̂)으로 같은 식 — 예측 불확실성만큼 더 헤지
- LGBM 급락      : 급락 확률이 학습 구간 평소 비율보다 높은 만큼 헤지
- LGBM 결합      : LGBM 변동성과 LGBM 급락의 평균
예측이 없는 날(2021년 이전)은 NaN 이다. 매개변수는 미리 정한 값이며 시험 결과로 고르지 않는다.
"""

from __future__ import annotations

import pandas as pd

from hedging.features.hedge_features import H
from hedging.models.lgbm_hedge import fold_base_rate, h_from_crash, h_from_vol, sigma_star
from hedging.strategies.hedge_rules import vol_target

PRED_FILES = {"vol": "y_logvol", "q80": "y_logvol_q80", "crash": "y_crash", "har": "har"}


def load_preds(folder: str = "data/processed") -> dict[str, pd.DataFrame]:
    return {k: pd.read_parquet(f"{folder}/ch08_pred_{v}.parquet") for k, v in PRED_FILES.items()}


def ml_hedge_ratios(df: pd.DataFrame, code: str, panel: pd.DataFrame, preds: dict[str, pd.DataFrame],
                    seed: int | None = None, full_at: float = 2.0) -> pd.DataFrame:
    """df(load_hedge_frame) 와 같은 index 의 정책별 h. seed=None 이면 시드 평균 예측을 쓴다."""
    col = "pred" if seed is None else f"pred_s{seed}"
    out = df[["date"]].copy()
    pc = panel[panel["code"] == code][["date", "fwd_vol"]]
    out = out.merge(pc, on="date", how="left")
    for k, p in preds.items():
        c = col if col in p.columns else "pred"  # HAR 은 시드가 없다
        out = out.merge(p.loc[p["code"] == code, ["date", c]].rename(columns={c: k}), on="date", how="left")
    s_star = sigma_star(out["fwd_vol"], H)
    rates = {y: fold_base_rate(panel, "y_crash", y) for y in out["date"].dt.year.unique()}
    base = out["date"].dt.year.map(rates)
    h = pd.DataFrame(index=df.index)
    h["규칙(변동성 목표)"] = vol_target(df).values
    h["HAR 변동성"] = h_from_vol(out["har"], s_star).values
    h["LGBM 변동성"] = h_from_vol(out["vol"], s_star).values
    h["LGBM 상단80%"] = h_from_vol(out["q80"], s_star).values
    h["LGBM 급락"] = h_from_crash(out["crash"], base, full_at).values
    h["LGBM 결합"] = (h["LGBM 변동성"] + h["LGBM 급락"]) / 2
    h.loc[out["vol"].isna().values, h.columns[1:]] = float("nan")
    return h
