"""주식선물 이론가격 (4장).

보유비용 모형(cost of carry), 단리·연 365일 기준:

    F* = S × (1 + r × τ) − D

- S: 현물가격, r: 무위험금리(기준금리 + spread), τ: 최종거래일까지 달력일수 / 365
- D: 오늘 이후 ~ 최종거래일 사이에 배당락이 있는 배당금 합계(주당)

배당락일: 결제가 T+2이므로, 배당기준일 이전 마지막 거래일의 '전 거래일'이 배당락일이다.

예상 배당(미래 참조 방지)
- 배당금은 대개 기준일이 지난 뒤 이사회에서 정해진다. 그 시점에는 금액을 모르므로
  '같은 종류(분기/결산)의 직전 배당금'을 예상값으로 쓴다(dividend_mode="expected").
- 실제 배당금을 쓰는 dividend_mode="actual"은 사후 비교용이다(미래 참조).
- 배당기준일 일정 자체(그날 배당이 있는지 없는지)는 미리 공시된다고 본다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedging.features.rates import base_rate


def ex_dividend_date(record_date: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """배당기준일 이전(포함) 마지막 거래일의 전 거래일. 달력 밖이면 평일 기준으로 근사한다."""
    if record_date > calendar[-1]:
        calendar = calendar.union(pd.bdate_range(calendar[-1], record_date + pd.Timedelta(days=7)))
    pos = calendar.searchsorted(record_date, side="right") - 1
    return calendar[pos - 1]


def dividend_table(divs: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """배당기준일별 배당락일, 실제 배당금, 예상 배당금, 상태(확정/없음/미정)."""
    d = divs.sort_values("record_date").reset_index(drop=True).copy()
    later_paid = d["dps"][::-1].cummax()[::-1].shift(-1).fillna(0) > 0  # 이후에 지급된 배당이 있는가
    d["status"] = np.where(d["dps"] > 0, "확정", np.where(later_paid, "없음", "미정"))
    d["ex_date"] = [ex_dividend_date(r, calendar) for r in d["record_date"]]

    expected = []
    for i, row in d.iterrows():
        if row["status"] == "없음":
            expected.append(0.0)
            continue
        prev = d.iloc[:i]
        prev = prev[prev["dps"] > 0]
        same = prev[prev["kind"] == row["kind"]]
        base = same if len(same) else prev
        expected.append(float(base["dps"].iloc[-1]) if len(base) else 0.0)
    d["expected_dps"] = expected
    d["actual_dps"] = np.where(d["status"] == "미정", np.nan, d["dps"])
    return d[["record_date", "ex_date", "kind", "status", "actual_dps", "expected_dps"]]


def fair_value(series: pd.DataFrame, divs: pd.DataFrame | None = None, spread: float = 0.0,
               dividend_mode: str = "expected") -> pd.DataFrame:
    """front_month_series()의 결과에 이론가격·괴리 열을 붙인다."""
    s = series.copy()
    s["rate"] = base_rate(s["date"]).values + spread
    s["tau"] = s["cal_dte"] / 365
    s["div"] = 0.0
    if divs is not None and len(divs):
        cal = pd.DatetimeIndex(s["date"])
        dt = dividend_table(divs, cal)
        col = "expected_dps" if dividend_mode == "expected" else "actual_dps"
        amounts = dt[col].fillna(dt["expected_dps"]).values
        ex = dt["ex_date"].values
        dates, ltds = s["date"].values, s["ltd"].values
        # 오늘 < 배당락일 <= 최종거래일 인 배당을 더한다
        mask = (ex[None, :] > dates[:, None]) & (ex[None, :] <= ltds[:, None])
        s["div"] = mask.astype(float) @ amounts
    s["fair"] = s["spot"] * (1 + s["rate"] * s["tau"]) - s["div"]
    s["theo_basis"] = s["fair"] - s["spot"]
    s["mispricing"] = s["front_close"] - s["fair"]
    s["mis_pct"] = s["mispricing"] / s["spot"] * 100
    s["theo_basis_pct"] = s["theo_basis"] / s["spot"] * 100
    # 시장이 가격에 반영한 연 조달금리 (만기가 너무 가까우면 불안정하므로 5일 이상만)
    s["implied_rate"] = np.where(s["cal_dte"] >= 5,
                                 ((s["front_close"] + s["div"]) / s["spot"] - 1) / s["tau"], np.nan)
    return s
