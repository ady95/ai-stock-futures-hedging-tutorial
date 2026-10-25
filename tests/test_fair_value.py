import numpy as np
import pandas as pd
import pytest

from hedging.features.basis import half_life
from hedging.features.fair_value import dividend_table, ex_dividend_date, fair_value
from hedging.features.rates import base_rate

CAL = pd.bdate_range("2025-12-01", "2026-10-30").difference(
    pd.to_datetime(["2025-12-25", "2025-12-31", "2026-09-24", "2026-09-25"]))


def test_ex_date_trading_record_date():
    # 기준일이 거래일이면 그 전 거래일이 배당락일 (T+2)
    assert ex_dividend_date(pd.Timestamp("2026-09-30"), CAL) == pd.Timestamp("2026-09-29")


def test_ex_date_year_end_holiday():
    # 12-31 휴장: 기준일 이전 마지막 거래일(12-30)의 전 거래일(12-29)
    assert ex_dividend_date(pd.Timestamp("2025-12-31"), CAL) == pd.Timestamp("2025-12-29")


def test_dividend_table_status_and_expectation():
    divs = pd.DataFrame({
        "record_date": pd.to_datetime(["2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"]),
        "dps": [566.0, 372.0, 0.0, 0.0],
        "kind": ["결산", "분기", "분기", "분기"],
    })
    dt = dividend_table(divs, CAL)
    assert dt["status"].tolist() == ["확정", "확정", "미정", "미정"]
    # 같은 종류의 직전 배당, 없으면 종류와 무관한 직전 배당(03-31 분기 → 직전 결산 566)
    assert dt["expected_dps"].tolist() == [0.0, 566.0, 372.0, 372.0]


def test_dividend_table_zero_before_later_payment_is_no_dividend():
    divs = pd.DataFrame({"record_date": pd.to_datetime(["2025-12-31", "2026-02-28"]),
                         "dps": [0.0, 1875.0], "kind": ["결산", "결산"]})
    dt = dividend_table(divs, CAL)
    assert dt["status"].tolist() == ["없음", "확정"]


def test_base_rate_lookup():
    r = base_rate(pd.to_datetime(["2026-07-15", "2026-07-16", "2026-10-23"]))
    assert r.tolist() == pytest.approx([0.025, 0.0275, 0.03])


def test_fair_value_carry_and_dividend():
    series = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-14", "2026-09-29"]),
        "spot": [250_000.0, 250_000.0], "front_close": [249_000.0, 250_500.0],
        "ltd": pd.to_datetime(["2026-10-08", "2026-10-08"]), "cal_dte": [24, 9],
    })
    divs = pd.DataFrame({"record_date": pd.to_datetime(["2026-06-30", "2026-09-30"]),
                         "dps": [374.0, 0.0], "kind": ["분기", "분기"]})
    fv = fair_value(series, divs)
    r = 0.03
    assert fv.loc[0, "div"] == 374.0  # 배당락(09-29)이 만기 전 → 예상 배당 차감
    assert fv.loc[1, "div"] == 0.0  # 배당락 당일에는 현물이 이미 배당락 가격
    assert fv.loc[0, "fair"] == pytest.approx(250_000 * (1 + r * 24 / 365) - 374)


def test_half_life_recovers_phi():
    rng = np.random.default_rng(0)
    x = [0.0]
    for _ in range(5000):
        x.append(0.5 * x[-1] + rng.normal())
    phi, hl = half_life(pd.Series(x), pd.Series(["A"] * len(x)))
    assert phi == pytest.approx(0.5, abs=0.03) and hl == pytest.approx(1.0, abs=0.1)
