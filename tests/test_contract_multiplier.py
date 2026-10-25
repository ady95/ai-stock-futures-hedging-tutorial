from datetime import date

import pytest

from hedging.instruments import (
    MULTIPLIER,
    last_trading_day,
    listed_months,
    price_limits,
    second_thursday,
    tick_size,
)


def test_multiplier():
    assert MULTIPLIER == 10


@pytest.mark.parametrize(
    "price,tick",
    [(1_999, 1), (2_000, 5), (19_990, 10), (20_000, 50), (199_900, 100), (200_000, 500), (500_000, 1_000)],
)
def test_tick_size(price, tick):
    assert tick_size(price) == tick


# KIS Open API 실측값 (2026-10-23 장 마감 기준 데이터, 2026-10-25 조회)
@pytest.mark.parametrize(
    "base,lo,hi",
    [
        (213_500, 192_200, 234_500),  # 삼성전자 202611
        (214_500, 193_100, 235_500),  # 삼성전자 202612
        (1_226_000, 1_104_000, 1_348_000),  # SK하이닉스 202611
    ],
)
def test_price_limits_match_kis(base, lo, hi):
    assert price_limits(base, 0.10) == (lo, hi)


def test_second_thursday():
    assert second_thursday(2026, 11) == date(2026, 11, 12)
    assert second_thursday(2026, 10) == date(2026, 10, 8)
    assert second_thursday(2027, 1) == date(2027, 1, 14)


def test_last_trading_day_holiday_moves_earlier():
    assert last_trading_day(2026, 11, holidays=[date(2026, 11, 12)]) == date(2026, 11, 11)


def test_listed_months_match_kis_master():
    # KIS 종목 마스터(2026-10-23 파일)의 주식선물 결제월
    assert listed_months(date(2026, 10, 25)) == [
        (2026, 11), (2026, 12), (2027, 1), (2027, 3), (2027, 6), (2027, 9)
    ]


def test_listed_months_roll_after_expiry():
    on_ltd = listed_months(date(2026, 11, 12))
    after = listed_months(date(2026, 11, 13))
    assert on_ltd[0] == (2026, 11)
    assert after == [(2026, 12), (2027, 1), (2027, 2), (2027, 3), (2027, 6), (2027, 9)]
