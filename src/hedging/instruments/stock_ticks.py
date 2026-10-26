"""국내 주식(현물) 호가가격단위 (5장 슬리피지 계산용).

2023-01-25 개편 이후 유가·코스닥 공통 7단계. 개편 이전 유가증권시장 단위는 달랐다.
출처: 한국거래소 호가가격단위 안내(공식 자료 기준, 미실측). 실제 매매 전 최신 규정 확인.
"""

from __future__ import annotations

import math

import pandas as pd

REFORM_DATE = pd.Timestamp("2023-01-25")
NEW_TABLE = [(2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500), (math.inf, 1_000)]
OLD_KOSPI_TABLE = [(1_000, 1), (5_000, 5), (10_000, 10), (50_000, 50), (100_000, 100), (500_000, 500),
                   (math.inf, 1_000)]


def stock_tick(price: float, day: pd.Timestamp | None = None) -> int:
    table = OLD_KOSPI_TABLE if day is not None and pd.Timestamp(day) < REFORM_DATE else NEW_TABLE
    for upper, tick in table:
        if price < upper:
            return tick
    raise ValueError(price)
