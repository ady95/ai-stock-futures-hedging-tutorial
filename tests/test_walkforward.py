import pandas as pd

from hedging.backtest.walkforward import walk_forward


def _pool(rows):
    return pd.DataFrame(rows, columns=["rule", "entry_date", "net_bp"]).assign(
        entry_date=lambda d: pd.to_datetime(d["entry_date"]))


def test_selection_uses_only_past_years():
    rows = []
    # A: 2020~2021엔 좋고 2022엔 나쁨, B: 2020~2021엔 나쁘고 2022엔 아주 좋음
    rows += [("A", f"202{y}-03-01", 10.0) for y in (0, 1) for _ in range(15)]
    rows += [("B", f"202{y}-03-01", -10.0) for y in (0, 1) for _ in range(15)]
    rows += [("A", "2022-03-01", -50.0)] * 5 + [("B", "2022-03-01", 100.0)] * 5
    wf, oos = walk_forward(_pool(rows), [2022], min_trades=20)
    assert wf.iloc[0]["선택 규칙"] == "A"  # 2022년의 B 성과를 미리 볼 수 없다
    assert oos["net_bp"].mean() == -50.0


def test_no_trade_when_nothing_positive():
    rows = [("A", "2021-03-01", -5.0)] * 30
    wf, oos = walk_forward(_pool(rows + [("A", "2022-03-01", 20.0)] * 3), [2022], min_trades=20)
    assert wf.iloc[0]["선택 규칙"] == "무거래" and oos.empty


def test_min_trades_required():
    rows = [("A", "2021-03-01", 50.0)] * 5  # 이익이지만 표본 부족
    wf, _ = walk_forward(_pool(rows), [2022], min_trades=20)
    assert wf.iloc[0]["선택 규칙"] == "무거래"
