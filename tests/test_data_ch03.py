from datetime import date

import pandas as pd
import pytest

from hedging.data.codes import futures_code, months_between, underlying_id
from hedging.data.dataset import front_month_series, valid_from
from hedging.data.kis import KisClient


# KIS Open API 실측으로 확인한 코드 (2026-10)
@pytest.mark.parametrize(
    "uid,y,m,code",
    [("11", 2026, 11, "A11611"), ("50", 2026, 11, "A50611"), ("11", 2026, 1, "A11601"),
     ("11", 2025, 12, "111W12"), ("50", 2025, 12, "150W12"), ("11", 2025, 10, "111W10"),
     ("11", 2024, 12, "111V12"), ("11", 2021, 3, "111R03"), ("11", 2020, 1, "111Q01")],
)
def test_futures_code(uid, y, m, code):
    assert futures_code(uid, y, m) == code
    assert underlying_id(code) == uid


def test_unknown_year_letter_raises():
    with pytest.raises(ValueError):
        futures_code("11", 2019, 12)


def test_months_between():
    assert months_between((2025, 11), (2026, 2)) == [(2025, 11), (2025, 12), (2026, 1), (2026, 2)]


def test_paged_daily_walks_backwards():
    # 100행씩만 주는 가짜 API로 페이지 조회가 끝까지 이어지는지 확인
    days = pd.bdate_range("2025-01-01", periods=250).strftime("%Y%m%d").tolist()

    def fetch(d1, d2):
        sel = [d for d in days if d1 <= d <= d2]
        return [{"stck_bsop_date": d} for d in sel[-100:]]

    rows = KisClient.__new__(KisClient)._paged_daily(fetch, "20250101", "20261231")
    assert sorted(r["stck_bsop_date"] for r in rows) == days


def _toy():
    cal = pd.bdate_range("2026-01-01", "2026-03-20")
    spot = pd.DataFrame({"date": cal, "close": 100.0})
    rows = []
    for code, ym, start, end in [("A11601", "202601", "2025-11-01", "2026-01-08"),
                                 ("A11602", "202602", "2025-12-01", "2026-02-12"),
                                 ("A11603", "202603", "2025-06-01", "2026-03-12")]:
        for d in cal[(cal >= start) & (cal <= end)]:
            rows.append({"code": code, "expiry_ym": ym, "date": d, "close": 101.0, "volume": 10})
    return pd.DataFrame(rows), spot


def test_valid_from_after_previous_expiry():
    fut, _ = _toy()
    assert valid_from(fut) == pd.Timestamp("2025-12-12")  # 2025-12물 최종거래일(12-11) 다음 날


def test_front_month_rolls_after_ltd():
    fut, spot = _toy()
    s = front_month_series(fut, spot, roll_days=0, today=date(2026, 3, 20))
    front = s.set_index("date")["front_code"]
    assert front[pd.Timestamp("2026-01-08")] == "A11601"
    assert front[pd.Timestamp("2026-01-09")] == "A11602"
    assert front[pd.Timestamp("2026-02-13")] == "A11603"
    assert int(s["roll"].sum()) == 2


def test_roll_days_moves_roll_earlier():
    fut, spot = _toy()
    s = front_month_series(fut, spot, roll_days=2, today=date(2026, 3, 20))
    front = s.set_index("date")["front_code"]
    assert front[pd.Timestamp("2026-01-06")] == "A11601"  # dte=2 → 아직 최근월물
    assert front[pd.Timestamp("2026-01-07")] == "A11602"  # dte=1 < 2 → 최종거래일 전날 교체
