import pandas as pd
import pytest

from hedging.data.krx import kis_short_code, universe, yearly_counts


# KRX 응답의 ISU_CD 와 KIS 단축코드 (2026-10 실측)
@pytest.mark.parametrize("isu,short", [("A116B000", "A11611"), ("A116C000", "A11612"), ("A1171000", "A11701"),
                                       ("111Q1000", "111Q01"), ("A116A000", "A11610"), ("ADT6B000", "ADT611")])
def test_kis_short_code(isu, short):
    assert kis_short_code(isu) == short


def test_universe_marks_delisted_and_etf():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2026-10-23", "2020-01-02", "2026-10-23"]),
        "market": ["kospi"] * 4,
        "und_name": ["삼성전자", "삼성전자", "사라진종목", "ETF 예시"],
    })
    u = universe(df).set_index("und_name")
    assert u.loc["삼성전자", "listed_now"] and not u.loc["사라진종목", "listed_now"]
    assert u.loc["ETF 예시", "is_etf"]
    yc = yearly_counts(df)
    assert yc.loc[2020, "kospi"] == 2 and yc.loc[2026, "kospi"] == 1


def test_listed_on_excludes_etf_and_other_days():
    from hedging.data.krx import listed_on

    df = pd.DataFrame({"date": pd.to_datetime(["2026-10-23", "2026-10-23", "2026-10-22"]),
                       "und_name": ["삼성전자", "ETF 예시", "어제만"]})
    assert listed_on(df, "2026-10-23") == ["삼성전자"]
