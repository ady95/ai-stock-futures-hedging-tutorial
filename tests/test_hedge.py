import numpy as np
import pandas as pd
import pytest

from hedging.backtest.hedge_ledger import HedgeConfig, hedge_metrics, run_hedge, target_contracts
from hedging.strategies.hedge_rules import ma_trend, vol_target

NOCOST = HedgeConfig(fut_fee=0, slip_ticks=0)


def _toy(spot, basis=0.0, ltd_idx=None):
    n = len(spot)
    dates = pd.bdate_range("2026-01-05", periods=n)
    ltd = dates[ltd_idx] if ltd_idx is not None else dates[-1] + pd.Timedelta(days=30)
    fut = np.array(spot, dtype=float) + basis
    df = pd.DataFrame({"date": dates, "spot": spot, "front_code": "A11603", "front_close": fut,
                       "front_settle": fut, "next_code": "A11606", "next_close": fut + 100, "ltd": ltd})
    if ltd_idx is not None:  # 최종거래일 다음 날부터는 차월물이 최근월물
        after = df.index > ltd_idx
        df.loc[after, "front_code"] = "A11606"
        px = df.loc[after, "spot"] + basis + 100
        df.loc[after, "front_close"] = px
        df.loc[after, "front_settle"] = px
        df.loc[after, "ltd"] = dates[-1] + pd.Timedelta(days=60)
    return df


def test_target_contracts_one_percent_steps():
    cfg = HedgeConfig()
    assert [target_contracts(h, cfg) for h in (0, 0.5, 0.505, 1.0, 1.2)] == [0, 50, 51, 100, 100]


def test_unhedged_equals_spot():
    df = _toy([100, 103, 99, 105])
    res = run_hedge(df, pd.Series(0.0, index=df.index), NOCOST)
    assert res["pnl"].sum() == pytest.approx(1000 * 5)


def test_full_hedge_zero_basis_change_is_flat():
    df = _toy([100, 103, 99, 105], basis=50)
    res = run_hedge(df, pd.Series(1.0, index=df.index), NOCOST)
    assert res["pnl"].sum() == pytest.approx(0)


def test_roll_on_last_trading_day_settles_on_spot():
    # 최종거래일(2번째 날)에 만기 월물은 현물 종가로 정산되고 차월물로 갈아탄다
    df = _toy([100, 102, 104, 106], basis=10, ltd_idx=1)
    res = run_hedge(df, pd.Series(1.0, index=df.index), NOCOST)
    assert bool(res.loc[1, "roll"])
    # 0→1일: 선물 110→102(최종결제=현물) 매도 이익 +8, 현물 +2 → 1,000주 × 10
    assert res.loc[1, "pnl"] == pytest.approx(1000 * (2 + 8))
    # 1→2일: 차월물을 next_close(=102+10+100=212)에 매도, 다음 날 정산가 104+10+100=214 → 선물 −2, 현물 +2
    assert res.loc[2, "pnl"] == pytest.approx(0)
    assert res["contracts"].iloc[-1] == 100


def test_costs_and_band():
    df = _toy([100] * 6)
    h = pd.Series([0.5, 0.52, 0.55, 0.62, 0.62, 0.62], index=df.index)
    cfg = HedgeConfig(fut_fee=0, slip_ticks=1, band=10)
    res = run_hedge(df, h, cfg)
    assert res["traded"].tolist() == [50, 0, 0, 12, 0, 0]  # 2·5계약 변화는 무시, 12계약에서 거래


def test_policies_use_only_past_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"spot": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))})
    full = vol_target(df)
    cut = vol_target(df.iloc[:200])
    assert np.allclose(full.iloc[:200].values, cut.values)  # 뒤 데이터를 붙여도 과거 h는 그대로
    assert np.allclose(ma_trend(df).iloc[:200].values, ma_trend(df.iloc[:200]).values)


def test_metrics_keys():
    df = _toy(list(100 + np.arange(30)))
    m = hedge_metrics(run_hedge(df, pd.Series(0.5, index=df.index)))
    assert {"평균 h", "연변동성%", "최대낙폭%", "헤지효과(분산감소)"} <= set(m)
