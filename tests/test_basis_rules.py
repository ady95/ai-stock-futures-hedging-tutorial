from dataclasses import replace

import pandas as pd
import pytest

from hedging.strategies.basis_rules import BasisRuleParams, order_candidates, run

ZERO = BasisRuleParams(entry_pct=0.5, exit_pct=0.0, min_dte=3, exec="close", qty=1000,
                       spot_fee=0, sell_tax=0, fut_fee=0, slip_ticks=0, margin_rate=0)


def _toy(mis, spot, fut):
    dates = pd.bdate_range("2026-03-02", periods=len(mis))
    n = len(mis)
    df = pd.DataFrame({
        "date": dates, "front_code": "A11603", "ltd": dates[-1], "usable": True, "div": 0.0,
        "mis_pct": mis, "dte": list(range(n - 1, -1, -1)), "spot": spot, "front_close": fut, "rate": 0.0,
        "front_volume": 1_000_000,
    })
    df["next_date"] = df["date"].shift(-1)
    df["next_spot_open"] = df["spot"].shift(-1)
    df["next_front_open_same"] = df["front_close"].shift(-1)
    return df


def test_enter_and_converge_at_close():
    df = _toy([0.6, 0.3, -0.1, 0.2, 0.1, 0.0], [100, 101, 102, 103, 104, 105], [101, 101.5, 101.9, 103, 104, 105])
    t = run(df, ZERO)
    assert len(t) == 1 and t.iloc[0]["reason"] == "수렴"
    # 현물 100→102 (+2), 선물 101→101.9 매도 (−0.9) → 1.1 × 1000주
    assert t.iloc[0]["gross"] == pytest.approx(1100)
    assert t.iloc[0]["net"] == pytest.approx(1100)


def test_hold_to_expiry_settles_on_spot_close():
    df = _toy([0.6, 0.4, 0.3, 0.2, 0.1, 0.05], [100, 99, 98, 97, 96, 95], [101, 100, 99, 98, 97, 96])
    t = run(df, ZERO)
    r = t.iloc[0]
    assert r["reason"] == "만기"
    # 만기: 선물은 현물 종가(95)로 정산 → 현물 −5 + 선물 (101−95)=+6 → +1 × 1000
    assert r["gross"] == pytest.approx(1000)


def test_next_open_enters_day_after_signal():
    df = _toy([0.6, 0.3, -0.1, 0.2, 0.1, 0.0], [100, 101, 102, 103, 104, 105], [101, 101.5, 101.9, 103, 104, 105])
    t = run(df, replace(ZERO, exec="next_open"))
    r = t.iloc[0]
    assert r["entry_date"] == df["date"][1] and r["exit_date"] == df["date"][3]


def test_costs_reduce_net():
    df = _toy([0.6, 0.3, -0.1, 0.2, 0.1, 0.0], [100, 101, 102, 103, 104, 105], [101, 101.5, 101.9, 103, 104, 105])
    t = run(df, replace(ZERO, sell_tax=0.002))
    assert t.iloc[0]["net"] == pytest.approx(1100 - 0.002 * 102 * 1000)


def test_dividend_window_blocks_entry():
    df = _toy([0.6, 0.3, -0.1, 0.2, 0.1, 0.0], [100] * 6, [101] * 6)
    df["div"] = 1.0
    assert run(df, ZERO).empty


def test_order_candidate_has_two_legs():
    df = _toy([0.6, 0.3, -0.1, 0.2, 0.1, 0.0], [100] * 6, [101] * 6)
    c = order_candidates(df, ZERO, "005930", df["date"][0])[0]
    assert c["action"] == "ENTER_CANDIDATE"
    assert [(leg["side"], leg["qty"]) for leg in c["legs"]] == [("BUY", 1000), ("SELL", 100)]
