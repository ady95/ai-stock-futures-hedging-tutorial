"""동적 헤지 백테스터 (7장, 8~9장 공용).

현물 qty 주를 계속 보유하고, 주식선물 매도 계약 수만 헤지 비율 h 에 맞춰 조정한다.

하루의 처리
1) 현물 손익 = qty × (S_t − S_{t−1})
2) 선물 손익 = −N × 승수 × (P_t − P_{t−1}), P 는 보유 월물의 정산가. 최종거래일에는 최종결제(현물 종가)
3) 조정: 목표 계약 수 N* = round(qty × h / 승수). |N* − N| ≥ band 일 때만 거래
   - exec="close": 그날 종가까지 보고 정한 h 를 그날 선물 종가(15:35~15:45 단일가)에 체결
   - exec="lag1" : 하루 늦게 실행(전날 정한 h 를 오늘 종가에) — 민감도
4) 롤오버: 최종거래일에 만기 월물은 최종결제로 정리, 같은 날 차월물 종가로 N* 매도
비용 = 체결 계약 × 승수 × (수수료율 × 가격 + 슬리피지 틱 × 호가단위). 최종결제는 슬리피지 없음.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedging.instruments.contracts import tick_size


@dataclass(frozen=True)
class HedgeConfig:
    qty: int = 1_000
    multiplier: int = 10
    fut_fee: float = 0.00003
    slip_ticks: float = 0.5
    band: int = 0  # 재조정 최소 계약 수 (0이면 매일 목표로 맞춤)
    exec: str = "close"  # "close" | "lag1"
    margin_rate: float = 0.36


def target_contracts(h: float, cfg: HedgeConfig) -> int:
    return int(np.floor(cfg.qty * float(np.clip(h, 0, 1)) / cfg.multiplier + 0.5))


def run_hedge(df: pd.DataFrame, h: pd.Series, cfg: HedgeConfig = HedgeConfig()) -> pd.DataFrame:
    """df: 최근월물 시계열(date, spot, front_code, front_close, front_settle, next_code, next_close, ltd).
    h: 같은 index 의 목표 헤지 비율."""
    d = df.reset_index(drop=True)
    hs = pd.Series(np.asarray(h, dtype=float), index=d.index)
    if cfg.exec == "lag1":
        hs = hs.shift(1)
    hs = hs.fillna(0.0)
    m, q = cfg.multiplier, cfg.qty
    n, p_prev = 0, np.nan
    rows = []
    for i, r in d.iterrows():
        s_prev = d.at[i - 1, "spot"] if i > 0 else r["spot"]
        spot_pnl = q * (r["spot"] - s_prev)
        is_ltd = r["date"] >= r["ltd"]
        # 최근월물은 정산가 = 종가(3-2)이므로 평가와 조정 체결 모두 정산가로 한다
        settle = r["front_settle"] if not pd.isna(r.get("front_settle", np.nan)) else r["front_close"]
        p_now = r["spot"] if is_ltd else settle  # 최종거래일: 최종결제가격 = 현물 종가
        fut_pnl = -n * m * (p_now - p_prev) if n > 0 else 0.0
        n_star = target_contracts(hs.iat[i], cfg)
        cost, traded = 0.0, 0
        if is_ltd:  # 만기 월물은 최종결제로 정리하고, 차월물 종가로 새로 매도(롤오버)
            if n > 0:
                cost += n * m * cfg.fut_fee * p_now
                traded += n
            nxt = r["next_close"] if not pd.isna(r.get("next_close", np.nan)) else r["spot"]
            if n_star > 0:
                cost += n_star * m * (cfg.fut_fee * nxt + cfg.slip_ticks * tick_size(nxt))
                traded += n_star
            n, p_prev = n_star, nxt
        else:
            dn = n_star - n
            if dn != 0 and (abs(dn) >= cfg.band or n_star == 0):
                cost += abs(dn) * m * (cfg.fut_fee * p_now + cfg.slip_ticks * tick_size(p_now))
                traded += abs(dn)
                n = n_star
            p_prev = p_now
        rows.append({"date": r["date"], "spot": r["spot"], "h_target": hs.iat[i], "contracts": n,
                     "h_actual": n * m / q, "spot_pnl": spot_pnl, "fut_pnl": fut_pnl, "cost": cost,
                     "traded": traded, "margin": n * m * p_prev * cfg.margin_rate, "roll": bool(is_ltd)})
    out = pd.DataFrame(rows)
    out["pnl"] = out["spot_pnl"] + out["fut_pnl"] - out["cost"]
    out["base"] = q * out["spot"].shift(1).fillna(out["spot"].iloc[0])
    out["ret"] = out["pnl"] / out["base"]
    out["spot_ret"] = out["spot_pnl"] / out["base"]
    return out


def hedge_metrics(res: pd.DataFrame) -> dict:
    r, rs = res["ret"].iloc[1:], res["spot_ret"].iloc[1:]
    years = len(r) / 252
    value = 1 + r.cumsum()  # 시작 명목금액 대비 누적 손익(단리)
    dd = value - value.cummax()
    var5 = r.quantile(0.05)
    down = rs < 0
    return {
        "평균 h": round(float(res["h_actual"].mean()), 3),
        "연수익%": round(float(r.mean() * 252 * 100), 2),
        "연변동성%": round(float(r.std() * np.sqrt(252) * 100), 2),
        "최대낙폭%": round(float(dd.min() * 100), 2),
        "CVaR5%(일)": round(float(r[r <= var5].mean() * 100), 3),
        "최악일%": round(float(r.min() * 100), 2),
        "하락일 포착률": round(float(r[down].mean() / rs[down].mean()), 3),
        "헤지효과(분산감소)": round(float(1 - r.var() / rs.var()), 3),
        "연 거래 계약": round(float(res["traded"].sum() / years), 0),
        "연 비용bp": round(float(res["cost"].sum() / res["base"].mean() / years * 1e4), 1),
        "평균 증거금(만원)": round(float(res["margin"].mean() / 1e4), 0),
    }


def load_hedge_frame(code: str) -> pd.DataFrame:
    """6장 전체 종목 피처(KRX 선물 기반)에 현물 거래량·거래대금을 붙인다(VWAP 정책용)."""
    f = pd.read_parquet(f"data/processed/universe/{code}_features.parquet")
    s = pd.read_parquet(f"data/raw/kis/stock_daily/{code}.parquet")[["date", "volume", "value"]]
    f = f.merge(s.rename(columns={"volume": "spot_volume", "value": "spot_value"}), on="date", how="left")
    return f.sort_values("date").reset_index(drop=True)


FRONTIER_GRID = np.round(np.arange(0, 1.01, 0.1), 2)
COMPARE_KEYS = ["연수익%", "연변동성%", "최대낙폭%", "CVaR5%(일)", "최악일%"]


def fixed_frontier(df: pd.DataFrame, cfg: HedgeConfig = HedgeConfig(), grid=FRONTIER_GRID) -> pd.DataFrame:
    """고정 헤지 h=0~1 곡선 (7-4). 동적 정책은 같은 평균 h 의 점과 비교한다."""
    return pd.DataFrame([{"h": h, **hedge_metrics(run_hedge(df, pd.Series(h, index=df.index), cfg))} for h in grid])


def vs_fixed(m: dict, fr: pd.DataFrame) -> dict:
    """같은 평균 h 의 고정 헤지(보간) 대비 차이(동적 − 고정)와, 같은 변동성에서의 수익 차."""
    out = {k: round(m[k] - float(np.interp(m["평균 h"], fr["h"], fr[k])), 3) for k in COMPARE_KEYS}
    order = fr.sort_values("연변동성%")
    out["같은 변동성 수익 차"] = round(m["연수익%"] - float(np.interp(m["연변동성%"], order["연변동성%"], order["연수익%"])), 2)
    return out
