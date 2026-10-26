"""성과 지표·시장국면 분해·레그 체결 위험 (6장)."""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIT = 100_000_000  # 거래 손익을 '1억원어치 거래'로 환산할 때의 명목금액


def ledger_metrics(led: pd.DataFrame, bench: pd.Series) -> dict:
    eq = led.set_index("date")["equity"]
    years = (eq.index[-1] - eq.index[0]).days / 365
    ret = eq.pct_change().dropna()
    exc = (eq.pct_change() - bench.pct_change()).dropna()
    dd = eq / eq.cummax() - 1
    return {
        "기간(년)": round(years, 2),
        "최종 평가액": eq.iloc[-1],
        "현금 보유 시 평가액": bench.iloc[-1],
        "초과 손익(원)": eq.iloc[-1] - bench.iloc[-1],
        "연환산 수익률%": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100,
        "현금 연환산%": ((bench.iloc[-1] / bench.iloc[0]) ** (1 / years) - 1) * 100,
        "초과수익 샤프": exc.mean() / exc.std() * np.sqrt(252) if exc.std() > 0 else np.nan,
        "최대낙폭%": dd.min() * 100,
        "포지션 보유일 비율%": led["in_position"].mean() * 100,
        "추가증거금 일수": int(led["margin_call"].sum()),
        "연환산 변동성%": ret.std() * np.sqrt(252) * 100,
    }


def trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"거래": 0, "승률": np.nan, "평균bp": np.nan, "중앙bp": np.nan, "하위5%bp": np.nan,
                "1억 환산 합계(만원)": 0.0}
    return {
        "거래": len(t),
        "승률": round(float((t["net_bp"] > 0).mean()), 3),
        "평균bp": round(float(t["net_bp"].mean()), 1),
        "중앙bp": round(float(t["net_bp"].median()), 1),
        "하위5%bp": round(float(t["net_bp"].quantile(0.05)), 1),
        "1억 환산 합계(만원)": round(float((t["net_bp"] * UNIT / 1e4).sum()) / 1e4, 1),
    }


def regime_table(features: pd.DataFrame) -> pd.DataFrame:
    """날짜별 시장국면 이름표. 모두 그날까지의 정보만 쓴다."""
    f = features.sort_values("date").copy()
    ret = np.log(f["spot"]).diff()
    vol = ret.rolling(20, min_periods=15).std() * np.sqrt(252)
    f["vol_regime"] = np.where(vol > vol.expanding(60).median(), "고변동", "저변동")
    f["trend_regime"] = np.where(f["spot"] / f["spot"].shift(60) - 1 > 0, "60일 상승", "60일 하락")
    f["rate_regime"] = np.where(f["rate"] >= 0.02, "기준금리 2% 이상", "기준금리 2% 미만")
    f["month_type"] = np.where(f["front_expiry"].str[4:].isin(["03", "06", "09", "12"]), "분기월물", "비분기월물")
    return f[["date", "vol_regime", "trend_regime", "rate_regime", "month_type"]]


def legging_mc(trades: pd.DataFrame, features: pd.DataFrame, p_miss: float, n_sims: int = 500,
               seed: int = 0) -> pd.DataFrame:
    """레그 체결 위험: 확률 p_miss로 선물 레그가 하루 늦게(다음 거래일 시가) 체결된다고 가정.

    진입 때 선물을 하루 늦게 팔면 그 사이 현물만 들고 있는 셈이고, 청산 때 하루 늦게 되사면
    그 사이 선물 매도만 남는 셈이다. 만기 청산은 최종결제라 지연이 없다.
    거래별 순손익(bp)의 시뮬레이션 분포를 돌려준다.
    """
    rng = np.random.default_rng(seed)
    f = features.set_index("date")
    dates = f.index
    fo = f["front_open"]

    def next_open(day):
        i = dates.searchsorted(day, side="right")
        return fo.iloc[i] if i < len(dates) else np.nan

    base = trades["net_bp"].values
    d_entry = np.array([(next_open(d) - fe) for d, fe in zip(trades["entry_date"], trades["entry_fut"])])
    d_exit = np.array([0.0 if r == "만기" else (fx - next_open(d))
                       for d, fx, r in zip(trades["exit_date"], trades["exit_fut"], trades["reason"])])
    scale = trades["qty"].values / trades["notional"].values * 1e4  # 원/주 → bp
    d_entry, d_exit = np.nan_to_num(d_entry) * scale, np.nan_to_num(d_exit) * scale
    out = []
    for _ in range(n_sims):
        me = rng.random(len(base)) < p_miss
        mx = rng.random(len(base)) < p_miss
        out.append(base + me * d_entry + mx * d_exit)
    sims = np.vstack(out)
    return pd.DataFrame({"평균bp": sims.mean(axis=1), "하위5%bp": np.quantile(sims, 0.05, axis=1),
                         "최악bp": sims.min(axis=1)})
