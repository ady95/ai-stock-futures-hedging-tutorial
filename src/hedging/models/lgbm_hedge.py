"""LightGBM 워크포워드 학습·예측과 헤지 비율 변환 (8장).

워크포워드(연 단위, 확장 창)
- 시험 연도 y: y년의 모든 행을 예측한다
- 학습: label_end < y년 첫 거래일 − embargo 인 행만 (레이블이 시험 구간에 닿지 않게)
- 조기 종료용 검증: 학습 구간의 마지막 1년 (그 역시 시험 구간과 겹치지 않음)
시드: 같은 데이터·하이퍼파라미터로 seed만 바꿔 여러 번 학습한다(PRD: 5개 이상).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_PARAMS = {
    "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 100, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "n_jobs": 4,
}


def _fit(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], target: str, seed: int,
         extra: dict | None = None):
    obj = "binary" if target in ("y_crash", "y_up") else "regression"
    params = {**BASE_PARAMS, "objective": obj, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
              **(extra or {})}
    dtr = lgb.Dataset(train[features], train[target])
    dva = lgb.Dataset(valid[features], valid[target], reference=dtr)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def walk_forward(panel: pd.DataFrame, target: str, test_years: list[int], features: list[str],
                 seeds: list[int], embargo_days: int = 5, params: dict | None = None) -> tuple[pd.DataFrame, list]:
    """시험 연도별로 학습해 예측을 모은다. 반환: 예측 DataFrame(seed별 열 + 평균·표준편차), 모델 목록.
    params: 기본값 덮어쓰기(예: {"objective": "quantile", "alpha": 0.8} — 분위 회귀)."""
    data = panel.dropna(subset=features)
    out, models = [], []
    for y in test_years:
        start = pd.Timestamp(f"{y}-01-01")
        cutoff = start - pd.Timedelta(days=int(embargo_days * 7 / 5) + 1)
        train_all = data[(data["label_end"] < cutoff) & data[target].notna()]
        valid = train_all[train_all["date"] >= cutoff - pd.DateOffset(years=1)]
        train = train_all[train_all["date"] < cutoff - pd.DateOffset(years=1)]
        test = data[(data["date"] >= start) & (data["date"] < pd.Timestamp(f"{y + 1}-01-01"))].copy()
        preds = []
        for s in seeds:
            m = _fit(train, valid, features, target, s, params)
            models.append((y, s, m))
            preds.append(m.predict(test[features], num_iteration=m.best_iteration))
        P = np.vstack(preds)
        for i, s in enumerate(seeds):
            test[f"pred_s{s}"] = P[i]
        test["pred"] = P.mean(axis=0)
        test["pred_std"] = P.std(axis=0)
        test["test_year"] = y
        test["n_train"] = len(train)
        out.append(test)
    return pd.concat(out, ignore_index=True), models


def range_shift(panel: pd.DataFrame, features: list[str], test_years: list[int], embargo_days: int = 5) -> pd.DataFrame:
    """시험 연도마다, 시험 행 중 피처 값이 학습 구간의 [최솟값, 최댓값] 밖에 있는 비율.
    레이블 없이 계산하므로 시험 결과를 보기 전에 확인할 수 있다. 나무 모델은 범위 밖을 외삽하지 못한다."""
    rows = {}
    for y in test_years:
        cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=int(embargo_days * 7 / 5) + 1)
        tr = panel[panel["label_end"] < cutoff]
        te = panel[panel["date"].dt.year == y]
        rows[y] = {f: float(((te[f] < tr[f].min()) | (te[f] > tr[f].max()))[te[f].notna()].mean()) for f in features}
    return pd.DataFrame(rows).T


# ---- 예측 → 헤지 비율 -----------------------------------------------------------
def h_from_vol(pred_logvol: pd.Series, target_vol: pd.Series) -> pd.Series:
    """앞을 보는 변동성 목표: 예측 변동성이 기준보다 높은 만큼 헤지. h = 1 − σ* / σ̂."""
    return (1 - target_vol / np.exp(pred_logvol)).clip(0, 1)


def sigma_star(fwd_vol: pd.Series, horizon: int, min_history: int = 120) -> pd.Series:
    """기준 변동성 σ*: 그날까지 '확정된' 미래 H일 변동성(레이블)의 중앙값(확장 창).
    t−H 행의 레이블은 t에 끝나므로 shift(H) 하면 t 시점에 알 수 있는 값만 쓴다.
    예측 대상과 같은 척도(H일 표본 변동성)라 h_from_vol의 기준으로 맞는다."""
    return fwd_vol.shift(horizon).expanding(min_history).median()


def fold_base_rate(panel: pd.DataFrame, target: str, year: int, embargo_days: int = 5) -> float:
    """시험 연도 year의 학습 구간(레이블이 cutoff 전에 끝난 행)에서 본 레이블 평균."""
    cutoff = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=int(embargo_days * 7 / 5) + 1)
    return float(panel.loc[panel["label_end"] < cutoff, target].mean())


def h_from_crash(p: pd.Series, base_rate: float, full_at: float = 2.0) -> pd.Series:
    """급락 확률이 평소(base_rate)보다 높아진 만큼 헤지. 평소 이하 → 0, 평소의 full_at 배 이상 → 1."""
    return ((p - base_rate) / (base_rate * (full_at - 1))).clip(0, 1)


def walk_forward_har(panel: pd.DataFrame, test_years: list[int], embargo_days: int = 5) -> pd.Series:
    """변동성 예측의 단순 기준선: log(미래 변동성) ~ log(vol_5), log(vol_20), log(vol_60) 선형 회귀(HAR 방식).
    LightGBM과 같은 워크포워드 구간으로 학습한다."""
    cols = ["vol_5", "vol_20", "vol_60"]
    data = panel.dropna(subset=cols)
    data = data[(data[cols] > 0).all(axis=1)]
    out = pd.Series(np.nan, index=panel.index)
    for y in test_years:
        start = pd.Timestamp(f"{y}-01-01")
        cutoff = start - pd.Timedelta(days=int(embargo_days * 7 / 5) + 1)
        tr = data[(data["label_end"] < cutoff) & data["y_logvol"].notna()]
        te = data[(data["date"] >= start) & (data["date"] < pd.Timestamp(f"{y + 1}-01-01"))]
        X = np.column_stack([np.ones(len(tr)), np.log(tr[cols])])
        beta, *_ = np.linalg.lstsq(X, tr["y_logvol"].values, rcond=None)
        out.loc[te.index] = np.column_stack([np.ones(len(te)), np.log(te[cols])]) @ beta
    return out


def top_liquid_codes(feats: dict[str, pd.DataFrame], n: int = 20) -> list[str]:
    """분석 기간 내내 상장된 종목 중 최근월물 평균 거래량 상위 n (7-4와 같은 대상)."""
    last = max(x["date"].max() for x in feats.values())
    full = {c: f for c, f in feats.items() if f["date"].min() <= pd.Timestamp("2019-12-31") and f["date"].max() == last}
    return sorted(full, key=lambda c: full[c]["front_volume"].mean(), reverse=True)[:n]
