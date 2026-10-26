from dataclasses import replace

import pandas as pd
import pytest

from hedging.backtest.ledger import LedgerConfig, run_ledger
from hedging.strategies.basis_rules import BasisRuleParams, run

P = BasisRuleParams(entry_pct=0.5, exit_pct=0.0, min_dte=3, exec="next_open", qty=1000,
                    spot_fee=0.00015, sell_tax=0.002, fut_fee=0.00003, slip_ticks=0.5)


def _toy(mis, spot, fut, spot_open=None, fut_open=None):
    dates = pd.bdate_range("2026-03-02", periods=len(mis))
    n = len(mis)
    df = pd.DataFrame({
        "date": dates, "front_code": "A11603", "ltd": dates[-1], "usable": True, "div": 0.0, "roll": False,
        "mis_pct": mis, "dte": list(range(n - 1, -1, -1)), "spot": spot, "front_close": fut,
        "front_settle": fut, "rate": 0.0, "front_volume": 1_000_000,
        "spot_open": spot_open or spot, "front_open": fut_open or fut,
    })
    df["next_date"] = df["date"].shift(-1)
    df["next_spot_open"] = df["spot_open"].shift(-1)
    df["next_front_open_same"] = df["front_open"].shift(-1)
    return df


@pytest.mark.parametrize("ex", ["close", "next_open"])
def test_ledger_matches_trade_level_engine(ex):
    df = _toy([0.6, 0.3, 0.2, -0.1, 0.2, 0.1, 0.0], [100_000 + 500 * i for i in range(7)],
              [100_700, 101_000, 101_400, 101_400, 102_200, 102_600, 103_000])
    p = replace(P, exec=ex)
    t5 = run(df, p)
    _, t6 = run_ledger(df, p, LedgerConfig(capital=1e9, fixed_qty=1000))
    assert len(t5) == len(t6) == 1
    assert t6.iloc[0]["entry_date"] == t5.iloc[0]["entry_date"]
    assert t6.iloc[0]["exit_date"] == t5.iloc[0]["exit_date"]
    assert t6.iloc[0]["gross"] == pytest.approx(t5.iloc[0]["gross"])
    assert t6.iloc[0]["costs"] == pytest.approx(t5.iloc[0]["fees_tax"] + t5.iloc[0]["slippage"])


def test_equity_identity_and_flat_end():
    df = _toy([0.6, 0.3, 0.2, 0.2, 0.2, 0.1, 0.0], [100_000] * 7, [100_700] * 6 + [100_000])
    led, tr = run_ledger(df, P, LedgerConfig(capital=1e9, fixed_qty=1000))
    last = led.iloc[-1]
    assert not last["in_position"] and last["fut_account"] == 0
    assert last["equity"] - 1e9 == pytest.approx(tr["net"].sum())  # 이자 0일 때 손익 = 거래 순손익 합


def test_margin_call_then_forced_liquidation():
    # 진입 후 선물이 매일 10%씩 오르고, 자본은 현물 + 위탁증거금에 겨우 맞춤
    spot = [100_000, 100_000, 110_000, 121_000, 133_100, 133_100, 133_100]
    fut = [100_700, 100_700, 110_700, 121_700, 133_800, 133_800, 133_800]
    df = _toy([0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], spot, fut)
    capital = 100_000 * 1000 + 100_700 * 1000 * 0.36 + 1_000_000
    led, tr = run_ledger(df, P, LedgerConfig(capital=capital, fixed_qty=1000))
    assert led["margin_call"].any()
    assert tr.iloc[0]["reason"] == "강제청산"


def test_entry_size_limited_by_cash():
    df = _toy([0.6, 0.3, 0.2, -0.1, 0.2, 0.1, 0.0], [100_000] * 7, [100_700] * 7)
    led, tr = run_ledger(df, P, LedgerConfig(capital=70_000_000, target_notional=100_000_000))
    assert tr.iloc[0]["qty"] < 1000  # 1억 목표지만 7천만원으로는 다 못 산다
    assert (led["cash"] >= 0).all()
