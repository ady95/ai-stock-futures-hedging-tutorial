import numpy as np
import pandas as pd
import pytest

from hedging.features.hedge_features import FEATURES, H, build_panel, stock_features
from hedging.models.lgbm_hedge import fold_base_rate, h_from_crash, h_from_vol, range_shift, sigma_star, walk_forward


def _frame(n=400, seed=0, code_shift=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    spot = 50_000 * np.exp(np.cumsum(rng.normal(code_shift, 0.02, n)))
    vol = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame({"date": dates, "spot": spot, "spot_volume": vol, "spot_value": vol * spot,
                         "basis_pct": rng.normal(0, 0.3, n), "mis_pct": rng.normal(0, 0.2, n),
                         "mis_z": rng.normal(0, 1, n), "dte": np.arange(n) % 40, "rate": 0.03})


def test_features_do_not_use_future_rows():
    df = _frame()
    full, cut = stock_features(df), stock_features(df.iloc[:300])
    cols = [c for c in FEATURES if c in cut.columns]
    assert np.allclose(full.loc[:299, cols].values, cut[cols].values, equal_nan=True)


def test_labels_are_future_and_end_empty():
    f = stock_features(_frame())
    i = 100
    assert f.at[i, "fwd_ret"] == pytest.approx(np.log(f.at[i + H, "spot"] / f.at[i, "spot"]))
    assert f.at[i, "label_end"] == f.at[i + H, "date"]
    assert f["y_crash"].tail(H).isna().all() and f["y_logvol"].tail(H).isna().all()
    assert f["y_crash"].head(20).isna().all()  # 20일 변동성(임계값)이 없으면 레이블도 비운다


def test_panel_market_features_use_same_day_only():
    frames = {"A": _frame(seed=1), "B": _frame(seed=2)}
    p = build_panel(frames)
    day = p["date"].unique()[50]
    cut = build_panel({k: v[v["date"] <= day] for k, v in frames.items()})
    a = p[p["date"] == day].sort_values("code")["mkt_ret_5"].values
    b = cut[cut["date"] == day].sort_values("code")["mkt_ret_5"].values
    assert np.allclose(a, b)


def test_walk_forward_train_labels_end_before_test():
    p = build_panel({"A": _frame(900, 1), "B": _frame(900, 2)})
    pred, models = walk_forward(p, "y_logvol", [2022], FEATURES, seeds=[1, 2])
    assert pred["date"].dt.year.eq(2022).all()
    assert {"pred_s1", "pred_s2", "pred", "pred_std"} <= set(pred.columns)
    assert len(models) == 2
    assert pred["n_train"].iloc[0] > 0


def test_h_mappings_bounds_and_direction():
    sig = pd.Series([0.2, 0.2, 0.2])
    assert h_from_vol(np.log(pd.Series([0.1, 0.2, 0.4])), sig).tolist() == pytest.approx([0, 0, 0.5])
    assert h_from_crash(pd.Series([0.05, 0.15, 0.30, 0.6]), 0.15).tolist() == pytest.approx([0, 0, 1, 1])
    assert h_from_crash(pd.Series([0.225]), 0.15).iloc[0] == pytest.approx(0.5)


def test_sigma_star_uses_only_finished_labels():
    fv = pd.Series(np.arange(200, dtype=float))
    s = sigma_star(fv, 5, min_history=10)
    # t=14 에는 0~9행 레이블(14일까지 끝남)만 쓴다
    assert np.isnan(s.iloc[13]) and s.iloc[14] == pytest.approx(np.median(np.arange(10)))


def test_range_shift_and_base_rate():
    p = build_panel({"A": _frame(900, 1)})
    p.loc[p["date"].dt.year == 2022, "rate"] = 0.05  # 학습 구간에 없던 값
    sh = range_shift(p, ["rate", "vol_20"], [2022])
    assert sh.loc[2022, "rate"] == 1.0
    br = fold_base_rate(p, "y_crash", 2022)
    assert 0 < br < 1
