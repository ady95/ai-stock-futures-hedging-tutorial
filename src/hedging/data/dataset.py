"""현물–주식선물 통합 일별 데이터셋과 품질 리포트 (3장).

최근월물 연속 시계열(front-month series)을 만든다.
- 거래일 달력은 현물 일봉의 날짜를 쓴다
- 각 월물의 최종거래일: 만기가 지난 월물은 실제 마지막 데이터 날짜, 진행 중인 월물은 두 번째 목요일 계산값
- roll_days: 최종거래일까지 남은 거래일(dte)이 이 값보다 작아지는 날 다음 월물로 넘어간다
  (0 = 최종거래일까지 보유, 1 = 최종거래일 당일 교체, 2 = 그 전날 교체)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from hedging.instruments.contracts import last_trading_day


def contract_table(fut: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """월물별 최종거래일(ltd)과 데이터 범위."""
    today = today or date.today()
    g = fut.groupby(["code", "expiry_ym"])["date"].agg(first="min", last="max").reset_index()

    def ltd(row):
        y, m = int(row.expiry_ym[:4]), int(row.expiry_ym[4:])
        planned = pd.Timestamp(last_trading_day(y, m))
        return row["last"] if planned.date() < today else planned

    g["ltd"] = g.apply(ltd, axis=1)
    g["planned_ltd"] = g["expiry_ym"].map(lambda s: pd.Timestamp(last_trading_day(int(s[:4]), int(s[4:]))))
    g["expired"] = g["planned_ltd"].dt.date < today
    return g.sort_values("ltd").reset_index(drop=True)


def valid_from(fut: pd.DataFrame) -> pd.Timestamp:
    """연속 시계열이 유효해지는 첫날.

    수집한 가장 이른 결제월보다 앞선 월물은 데이터가 없으므로, 그 직전 월물의 최종거래일까지는
    '진짜 최근월물'을 알 수 없다. 예) 2020-01물부터 수집 → 2019-12물 최종거래일 다음 날부터 유효.
    """
    first = fut["expiry_ym"].min()
    y, m = int(first[:4]), int(first[4:])
    y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return pd.Timestamp(last_trading_day(y, m)) + pd.Timedelta(days=1)


def front_month_series(
    fut: pd.DataFrame, spot: pd.DataFrame, roll_days: int = 0, today: date | None = None
) -> pd.DataFrame:
    spot = spot[spot["date"] >= valid_from(fut)]
    cal = pd.DatetimeIndex(sorted(spot["date"].unique()))
    ct = contract_table(fut, today)
    ct["ltd_pos"] = cal.searchsorted(ct["ltd"])
    # 최종거래일이 데이터 마지막 날보다 뒤면(진행 중인 월물) 남은 평일 수를 더해 위치를 추정한다
    beyond = ct["ltd"] > cal[-1]
    ct.loc[beyond, "ltd_pos"] = (len(cal) - 1) + np.busday_count(
        (cal[-1] + pd.Timedelta(days=1)).date(), (ct.loc[beyond, "ltd"] + pd.Timedelta(days=1)).dt.date.tolist())
    px = fut.set_index(["code", "date"])[["close", "volume"]]
    spot_px = spot.set_index("date")["close"]

    rows = []
    for pos, d in enumerate(cal):
        live = ct[(ct["first"] <= d) & (ct["ltd_pos"] - pos >= roll_days) & (ct["ltd"] >= d)]
        if live.empty:
            continue
        front = live.iloc[0]
        nxt = live.iloc[1] if len(live) > 1 else None
        rec = {"date": d, "spot": spot_px.get(d, np.nan), "front_code": front["code"],
               "front_expiry": front["expiry_ym"], "ltd": front["ltd"], "dte": int(front["ltd_pos"] - pos)}
        for tag, c in (("front", front), ("next", nxt)):
            if c is None:
                rec[f"{tag}_close"], rec[f"{tag}_volume"] = np.nan, np.nan
                continue
            if tag == "next":
                rec["next_code"] = c["code"]
            try:
                close, vol = px.loc[(c["code"], d)]
            except KeyError:
                close, vol = np.nan, np.nan
            rec[f"{tag}_close"], rec[f"{tag}_volume"] = close, vol
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["cal_dte"] = (df["ltd"] - df["date"]).dt.days
    df["basis"] = df["front_close"] - df["spot"]
    df["basis_pct"] = df["basis"] / df["spot"] * 100
    df["roll"] = df["front_code"].ne(df["front_code"].shift()) & df.index.to_series().gt(0)
    return df


def quality_report(series: pd.DataFrame, fut: pd.DataFrame, spot: pd.DataFrame,
                   outlier_pct: float = 3.0) -> dict:
    """데이터 품질 지표. 숫자가 0이 아니라고 무조건 오류는 아니며, 원인을 확인할 목록이다."""
    ct = contract_table(fut)
    expired = ct[ct["expired"]]
    shifted = expired[expired["ltd"] != expired["planned_ltd"]]
    on_ltd = series[series["date"] == series["ltd"]]
    conv = (on_ltd["front_close"] - on_ltd["spot"]).abs() / on_ltd["spot"] * 100
    out = series[series["basis_pct"].abs() > outlier_pct]
    return {
        "기간": f"{series['date'].min():%Y-%m-%d} ~ {series['date'].max():%Y-%m-%d}",
        "현물 거래일": len(spot),
        "통합 시계열 행": len(series),
        "수집 월물 수": fut["code"].nunique(),
        "최근월물 가격 결측일": int(series["front_close"].isna().sum()),
        "최근월물 거래량 0일": int((series["front_volume"] == 0).sum()),
        "차근월물 거래량 0 비율(%)": round(float((series["next_volume"] == 0).mean() * 100), 1),
        "롤오버 횟수": int(series["roll"].sum()),
        "날짜 중복(현물/선물)": (int(spot["date"].duplicated().sum()),
                          int(fut.duplicated(["code", "date"]).sum())),
        "최종거래일이 두 번째 목요일과 다른 월물": shifted[["code", "planned_ltd", "ltd"]].assign(
            planned_ltd=lambda x: x["planned_ltd"].dt.strftime("%Y-%m-%d"),
            ltd=lambda x: x["ltd"].dt.strftime("%Y-%m-%d")).values.tolist(),
        "만기일 |F-S|/S 최대(%)": round(float(conv.max()), 3) if len(conv) else None,
        f"|베이시스| > {outlier_pct}% 일수": len(out),
        "베이시스 이상치 상위": out.reindex(out["basis_pct"].abs().sort_values(ascending=False).index)
        .head(5)[["date", "front_code", "spot", "front_close", "basis_pct"]]
        .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"), basis_pct=lambda x: x["basis_pct"].round(2))
        .values.tolist(),
    }
