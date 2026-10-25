"""주식선물 단축코드 규칙 (3장).

KIS Open API 실측(2026-10)으로 확인한 두 가지 체계:

- 2026년 1월물 이후: "A" + 기초자산 번호(2자리) + 연도 끝자리 + 월(2자리)   예) A11611 = 삼성전자 2026-11
- 2025년 12월물 이전: "1" + 기초자산 번호(2자리) + 연도 문자 + 월(2자리)  예) 111W12 = 삼성전자 2025-12

연도 문자는 실제 조회로 확인한 2020~2025만 둔다. 확인되지 않은 연도는 오류를 낸다.
"""

from __future__ import annotations

NEW_SYSTEM_FROM = (2026, 1)
YEAR_LETTERS = {2020: "Q", 2021: "R", 2022: "S", 2023: "T", 2024: "V", 2025: "W"}


def underlying_id(short_code: str) -> str:
    """단축코드에서 기초자산 번호(2자리)를 꺼낸다. 예) A11611 → 11, 111W12 → 11"""
    if len(short_code) != 6 or short_code[0] not in "A1":
        raise ValueError(f"주식선물 단축코드가 아닙니다: {short_code}")
    return short_code[1:3]


def futures_code(und_id: str, year: int, month: int) -> str:
    """기초자산 번호와 결제 연월로 단축코드를 만든다."""
    if (year, month) >= NEW_SYSTEM_FROM:
        return f"A{und_id}{year % 10}{month:02d}"
    if year not in YEAR_LETTERS:
        raise ValueError(f"{year}년 옛 코드의 연도 문자는 확인되지 않았습니다")
    return f"1{und_id}{YEAR_LETTERS[year]}{month:02d}"


def months_between(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """(연, 월) 범위의 모든 달. 양 끝 포함."""
    y, m = start
    out = []
    while (y, m) <= end:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out
