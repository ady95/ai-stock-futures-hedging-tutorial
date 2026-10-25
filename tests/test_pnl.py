import numpy as np
import pytest

from hedging.pnl import ZERO_COST, CostModel, hindsight_pnl, hold_both_pnl, split_exit_pnl
from hedging.sim import gbm_paths, scenario_paths


@pytest.mark.parametrize("s1", [80_000, 100_000, 125_000])
def test_same_qty_price_pnl_is_zero(s1):
    r = hold_both_pnl(100_000, s1, 100, 20, ZERO_COST)
    assert r.gross == 0


def test_cost_breakdown():
    c = CostModel(fee_rate=0.00015, sell_tax_rate=0.002, borrow_rate=0.03)
    r = hold_both_pnl(100_000, 100_000, 100, 20, c)
    long_cost = 0.00015 * 200_000 * 100 + 0.002 * 100_000 * 100  # 23,000
    short_cost = long_cost + 0.03 * 100_000 * 100 * 20 / 365
    assert r.cost == pytest.approx(long_cost + short_cost)


def test_split_exit_trend_vs_reversal():
    paths = scenario_paths(100_000, 20)
    up = split_exit_pnl(paths["상승 지속"], 100, 0.05)
    rev = split_exit_pnl(paths["상승 후 복귀"], 100, 0.05)
    assert up.first_leg == "long" and not up.recovered and up.gross == pytest.approx(-500_000)
    assert rev.first_leg == "long" and rev.recovered and rev.gross > 0


def test_split_exit_mean_near_zero_without_cost():
    # 기대수익 0인 경로에서는 청산 규칙을 바꿔도 평균 가격손익이 0 근처다
    paths = gbm_paths(100_000, 0.0, 0.3, 60, 20_000, seed=7)
    g = np.array([split_exit_pnl(p, 100, 0.05).gross for p in paths])
    stderr = g.std() / np.sqrt(len(g))
    assert abs(g.mean()) < 3 * stderr


def test_hindsight_is_never_negative():
    paths = gbm_paths(100_000, 0.0, 0.3, 60, 500, seed=1)
    assert all(hindsight_pnl(p, 100).gross >= 0 for p in paths)


def test_gbm_reproducible():
    a = gbm_paths(100, 0, 0.2, 10, 3, seed=3)
    b = gbm_paths(100, 0, 0.2, 10, 3, seed=3)
    assert a.shape == (3, 11) and np.array_equal(a, b) and (a[:, 0] == 100).all()
