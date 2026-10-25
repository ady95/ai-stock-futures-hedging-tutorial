import pytest

from hedging.risk import (
    MarginRates,
    daily_settlement,
    initial_margin,
    margin_call_price,
    plan_hedge,
)

RATES = MarginRates(initial=0.36, maintenance=0.24)


@pytest.mark.parametrize(
    "qty,mode,n,net",
    [(203, "nearest", 20, 3), (203, "up", 21, -7), (205, "nearest", 21, -5), (9, "nearest", 1, -1), (4, "nearest", 0, 4)],
)
def test_plan_hedge_rounding(qty, mode, n, net):
    p = plan_hedge(qty, 1.0, 10, mode)
    assert p.contracts == n and p.net_shares == net


def test_plan_hedge_value_basis():
    # 금액 기준: 203주 × 213,500 / (10 × 215,000) = 20.16 → 20계약
    assert plan_hedge(203, 1.0, 10, "nearest", 213_500, 215_000).contracts == 20


def test_initial_margin():
    assert initial_margin(215_000, 20, RATES) == pytest.approx(15_480_000)


def test_margin_call_price_short_default_deposit():
    # 예탁금 = 위탁증거금일 때 매도 포지션은 F0 × 1.36 / 1.24 에서 마진콜
    f = margin_call_price(215_000, 20, RATES)
    assert f / 215_000 == pytest.approx(1.36 / 1.24)


def test_margin_call_price_more_cash_more_room():
    base = margin_call_price(215_000, 20, RATES)
    more = margin_call_price(215_000, 20, RATES, deposit=25_000_000)
    assert more > base


def test_daily_settlement_cashflow():
    df = daily_settlement([215_000, 236_500], 20, RATES, side="short")
    row = df.iloc[1]
    assert row["variation"] == pytest.approx(-21_500 * 200)
    assert bool(row["call"]) is True
    assert row["equity"] == pytest.approx(initial_margin(236_500, 20, RATES))  # 위탁 수준까지 채움
