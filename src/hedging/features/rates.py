"""무위험금리 근사: 한국은행 기준금리 변경 이력 (4장).

이론가격의 금리는 보통 CD(91일) 같은 단기 시장금리를 쓴다. 이 책은 누구나 확인할 수 있는
기준금리 이력에 스프레드를 더해 근사한다. 더 정확히 하려면 한국은행 ECOS의 CD 금리로 바꾸면 된다.

출처: 한국은행 금융통화위원회 통화정책방향 결정(공개 자료).
2026-10-22 결정은 집필 시점(2026-10-25)에 확인하지 못해 3.00% 유지로 가정한다.
"""

from __future__ import annotations

import pandas as pd

# (적용 시작일, 연 %)
BASE_RATE_CHANGES: list[tuple[str, float]] = [
    ("2019-07-18", 1.50),
    ("2019-10-16", 1.25),
    ("2020-03-17", 0.75),
    ("2020-05-28", 0.50),
    ("2021-08-26", 0.75),
    ("2021-11-25", 1.00),
    ("2022-01-14", 1.25),
    ("2022-04-14", 1.50),
    ("2022-05-26", 1.75),
    ("2022-07-13", 2.25),
    ("2022-08-25", 2.50),
    ("2022-10-12", 3.00),
    ("2022-11-24", 3.25),
    ("2023-01-13", 3.50),
    ("2024-10-11", 3.25),
    ("2024-11-28", 3.00),
    ("2025-02-25", 2.75),
    ("2025-05-29", 2.50),
    ("2026-07-16", 2.75),
    ("2026-08-27", 3.00),
]


def base_rate(dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """각 날짜에 적용되던 기준금리(소수, 예: 0.03)."""
    table = pd.Series([r / 100 for _, r in BASE_RATE_CHANGES],
                      index=pd.to_datetime([d for d, _ in BASE_RATE_CHANGES]))
    idx = pd.DatetimeIndex(dates)
    pos = table.index.searchsorted(idx, side="right") - 1
    if (pos < 0).any():
        raise ValueError("기준금리 이력보다 이른 날짜가 있습니다")
    return pd.Series(table.values[pos], index=idx)
