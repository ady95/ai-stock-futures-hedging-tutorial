"""국내 개별주식선물 계약 명세 (2장).

출처: 한국거래소 주식선물 상품 안내(Single Stock Futures) — 거래단위 10주,
결제월 비분기월 2개 + 분기월 4개, 최종거래일 결제월 두 번째 목요일,
호가가격단위는 선물가격 구간별, 가격제한폭 3단계(10%·20%·30%).
기업행사로 거래승수가 조정된 계약은 마스터 파일의 승수를 따라야 한다.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Iterable

MULTIPLIER = 10  # 거래단위(거래승수): 1계약 = 10주

# (가격 상한 미만, 호가가격단위)
TICK_TABLE: list[tuple[float, int]] = [
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
    (math.inf, 1_000),
]

QUARTER_MONTHS = (3, 6, 9, 12)


def tick_size(price: float) -> int:
    """선물가격 구간별 호가가격단위."""
    for upper, tick in TICK_TABLE:
        if price < upper:
            return tick
    raise ValueError(price)


def floor_to_tick(price: float) -> int:
    tick = tick_size(price)
    return int(math.floor(round(price / tick, 9)) * tick)


def ceil_to_tick(price: float) -> int:
    tick = tick_size(price)
    return int(math.ceil(round(price / tick, 9)) * tick)


def price_limits(base_price: float, pct: float = 0.10) -> tuple[int, int]:
    """기준가격 대비 가격제한폭의 (하한가, 상한가). 상한은 내림, 하한은 올림으로 호가단위에 맞춘다."""
    return ceil_to_tick(base_price * (1 - pct)), floor_to_tick(base_price * (1 + pct))


def second_thursday(year: int, month: int) -> date:
    d = date(year, month, 1)
    first_thu = d + timedelta(days=(3 - d.weekday()) % 7)  # weekday(): 월=0 … 목=3
    return first_thu + timedelta(days=7)


def last_trading_day(year: int, month: int, holidays: Iterable[date] = ()) -> date:
    """최종거래일: 결제월 두 번째 목요일. 휴장일이면 직전 거래일로 앞당긴다."""
    hol = set(holidays)
    d = second_thursday(year, month)
    while d.weekday() >= 5 or d in hol:
        d -= timedelta(days=1)
    return d


def _next_month(y: int, m: int) -> tuple[int, int]:
    return (y + 1, 1) if m == 12 else (y, m + 1)


def listed_months(today: date, holidays: Iterable[date] = ()) -> list[tuple[int, int]]:
    """오늘 거래 가능한 결제월 목록: 가까운 비분기월 2개 + 가까운 분기월 4개.

    이번 달 최종거래일이 지났으면 다음 달부터 센다.
    """
    hol = list(holidays)
    y, m = today.year, today.month
    if today > last_trading_day(y, m, hol):
        y, m = _next_month(y, m)
    quarterly: list[tuple[int, int]] = []
    non_quarterly: list[tuple[int, int]] = []
    while len(quarterly) < 4 or len(non_quarterly) < 2:
        if m in QUARTER_MONTHS:
            if len(quarterly) < 4:
                quarterly.append((y, m))
        elif len(non_quarterly) < 2:
            non_quarterly.append((y, m))
        y, m = _next_month(y, m)
    return sorted(quarterly + non_quarterly)


def remaining_days(today: date, year: int, month: int, holidays: Iterable[date] = ()) -> int:
    """최종거래일까지 남은 달력일수."""
    return (last_trading_day(year, month, holidays) - today).days
