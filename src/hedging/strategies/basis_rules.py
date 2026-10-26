"""규칙 기반 베이시스 전략: 선물이 이론가보다 비쌀 때 현물 매수 + 선물 매도 (5장).

한 방향만 다룬다. 반대 방향(선물 매수 + 현물 공매도)은 공매도가 필요하고 배당 위험이 커서 범위 밖이다.

진입: 그날 괴리율 ≥ entry_pct, 분석 사용일(usable), 잔존 거래일 ≥ min_dte,
      (옵션) 최종거래일 전에 배당락이 없을 것
청산: 괴리율 ≤ exit_pct 인 사용일 → 조기 청산 / 아니면 최종거래일에 만기 청산
      만기 청산은 선물이 최종결제가격(= 현물 종가)으로 정산되므로 두 레그 모두 현물 종가로 계산한다
체결: exec="close"     — 신호가 난 날 종가에 체결(낙관적 상한, 현실에서는 어렵다)
      exec="next_open" — 다음 거래일 시가에 체결(현실적)

모든 숫자(수수료·세금·슬리피지)는 예시 값이다. 5장의 거래별 손익은 표본 내 참고치이며
검증은 6장의 워크포워드로 한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from hedging.instruments.contracts import tick_size as futures_tick
from hedging.instruments.stock_ticks import stock_tick


@dataclass(frozen=True)
class BasisRuleParams:
    entry_pct: float = 0.5  # 진입 괴리율(%)
    exit_pct: float = 0.0  # 조기 청산 괴리율(%)
    min_dte: int = 5  # 진입 시 최종거래일까지 최소 거래일
    skip_dividend: bool = True  # 보유 중 배당락이 있으면 진입하지 않음
    exec: str = "next_open"  # "close" | "next_open"
    qty: int = 1_000  # 현물 주수 (선물 qty/10 계약)
    spot_fee: float = 0.00015  # 현물 수수료, 편도 (예시)
    sell_tax: float = 0.0020  # 현물 매도세 (예시)
    fut_fee: float = 0.00003  # 선물 수수료, 편도 (예시)
    slip_ticks: float = 0.5  # 레그·체결당 슬리피지(호가단위 배수). 0.5 = 반 틱(스프레드의 절반)
    margin_rate: float = 0.36  # 위탁증거금률 — 증거금 기회비용 계산용
    min_volume: float = 0  # 신호일 최근월물 최소 거래량(계약). 여러 종목으로 넓힐 때 쓴다(6장)


def attach_opens(fv: pd.DataFrame, fut_raw: pd.DataFrame, spot_raw: pd.DataFrame) -> pd.DataFrame:
    """최근월물 시가·현물 시가와, 다음 거래일의 같은 월물 시가를 붙인다."""
    s = fv.copy()
    fo = fut_raw.set_index(["code", "date"])["open"]
    so = spot_raw.set_index("date")["open"]
    s["front_open"] = [fo.get((c, d), np.nan) for c, d in zip(s["front_code"], s["date"])]
    s["spot_open"] = s["date"].map(so)
    nxt = s["date"].shift(-1)
    s["next_date"] = nxt
    s["next_spot_open"] = s["spot_open"].shift(-1)
    s["next_front_open_same"] = [fo.get((c, d), np.nan) for c, d in zip(s["front_code"], nxt)]
    s["next_rate"], s["next_div"] = s["rate"].shift(-1), s["div"].shift(-1)
    return s


def mis_at(spot: float, fut: float, rate: float, day: pd.Timestamp, ltd: pd.Timestamp) -> float:
    """체결 가격 기준 괴리율(%). 배당 구간은 진입하지 않으므로 배당은 0으로 본다."""
    tau = max((ltd - day).days, 0) / 365
    return (fut - spot * (1 + rate * tau)) / spot * 100


def _exec_prices(row: pd.Series, mode: str) -> tuple[pd.Timestamp, float, float]:
    if mode == "close":
        return row["date"], row["spot"], row["front_close"]
    return row["next_date"], row["next_spot_open"], row["next_front_open_same"]


def run(df: pd.DataFrame, p: BasisRuleParams) -> pd.DataFrame:
    """하루씩 진행하며 진입·청산 신호를 만들고 거래 목록을 돌려준다(한 번에 한 포지션)."""
    trades, pos = [], None
    rows = df.reset_index(drop=True)
    for i, r in rows.iterrows():
        if pos is None:
            ok = (r["usable"] and r["mis_pct"] >= p.entry_pct and r["dte"] >= p.min_dte
                  and not (p.skip_dividend and r["div"] > 0) and r["front_volume"] >= p.min_volume)
            if not ok:
                continue
            d, s_px, f_px = _exec_prices(r, p.exec)
            if pd.isna(s_px) or pd.isna(f_px):
                continue
            pos = {"code": r["front_code"], "ltd": r["ltd"], "signal_date": r["date"], "entry_date": d,
                   "entry_spot": s_px, "entry_fut": f_px, "signal_mis": r["mis_pct"], "rate": r["rate"],
                   "entry_mis": mis_at(s_px, f_px, r["rate"], d, r["ltd"])}
            continue
        # 포지션 보유 중
        if r["front_code"] != pos["code"]:  # 최종거래일 다음 날: 직전 행에서 만기 처리됐어야 함
            raise RuntimeError(f"만기 처리 누락: {pos['code']} {r['date']}")
        if r["date"] >= pos["ltd"]:  # 만기: 두 레그 모두 현물 종가
            trades.append(_close(pos, r["date"], r["spot"], r["spot"], "만기", p, expiry=True))
            pos = None
            continue
        if r["usable"] and r["mis_pct"] <= p.exit_pct and r["date"] > pos["entry_date"]:
            d, s_px, f_px = _exec_prices(r, p.exec)
            if pd.isna(s_px) or pd.isna(f_px) or (p.exec == "next_open" and d > pos["ltd"]):
                continue
            trades.append(_close(pos, d, s_px, f_px, "수렴", p, expiry=False))
            pos = None
    return pd.DataFrame(trades)


def _close(pos: dict, d: pd.Timestamp, s_px: float, f_px: float, reason: str, p: BasisRuleParams,
           expiry: bool) -> dict:
    q = p.qty
    days = (d - pos["entry_date"]).days
    s0, f0 = pos["entry_spot"], pos["entry_fut"]
    gross_spot = (s_px - s0) * q
    gross_fut = (f0 - f_px) * q
    financing = pos["rate"] * s0 * q * days / 365  # 현물 매수 자금의 조달(기회)비용
    margin_opp = pos["rate"] * p.margin_rate * f0 * q * days / 365  # 증거금으로 묶인 돈의 기회비용
    fees = p.spot_fee * (s0 + s_px) * q + p.fut_fee * (f0 + f_px) * q
    tax = p.sell_tax * s_px * q
    slip_ticks = stock_tick(s0, pos["entry_date"]) + stock_tick(s_px, d) + futures_tick(f0)
    if not expiry:
        slip_ticks += futures_tick(f_px)  # 만기 청산 시 선물은 최종결제가격으로 정산되어 슬리피지 없음
    slippage = p.slip_ticks * slip_ticks * q
    net = gross_spot + gross_fut - financing - margin_opp - fees - tax - slippage
    notional = s0 * q
    return {
        "code": pos["code"], "signal_date": pos["signal_date"], "entry_date": pos["entry_date"], "exit_date": d,
        "reason": reason, "days": days, "signal_mis_pct": pos["signal_mis"],
        "entry_mis_pct": pos["entry_mis"], "exit_mis_pct": mis_at(s_px, f_px, pos["rate"], d, pos["ltd"]),
        "entry_basis_pct": (f0 - s0) / s0 * 100, "gross": gross_spot + gross_fut,
        "financing": financing + margin_opp, "fees_tax": fees + tax, "slippage": slippage, "net": net,
        "net_bp": net / notional * 1e4, "notional": notional,
        "qty": q, "entry_spot": s0, "entry_fut": f0, "exit_spot": s_px, "exit_fut": f_px,
    }


def decompose(trades: pd.DataFrame) -> dict:
    """거래당 평균을 bp(현물 명목금액 대비)로 분해한다."""
    if trades.empty:
        return {}
    bp = lambda col: float((trades[col] / trades["notional"] * 1e4).mean())  # noqa: E731
    return {
        "신호 괴리%": float(trades["signal_mis_pct"].mean()),
        "진입 괴리%": float(trades["entry_mis_pct"].mean()),
        "청산 괴리%": float(trades["exit_mis_pct"].mean()),
        "가격손익bp": bp("gross"),
        "조달·증거금bp": -bp("financing"),
        "수수료·세금bp": -bp("fees_tax"),
        "슬리피지bp": -bp("slippage"),
        "순손익bp": bp("net"),
    }


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "win_rate": np.nan, "avg_net_bp": np.nan, "total_net": 0.0, "avg_days": np.nan}
    return {
        "trades": len(trades),
        "win_rate": float((trades["net"] > 0).mean()),
        "avg_net_bp": float(trades["net_bp"].mean()),
        "total_net": float(trades["net"].sum()),
        "avg_days": float(trades["days"].mean()),
        "expiry_share": float((trades["reason"] == "만기").mean()),
    }


def order_candidates(df: pd.DataFrame, p: BasisRuleParams, und_code: str, day: str | pd.Timestamp) -> list[dict]:
    """그날 종가 기준으로 '다음 거래일에 낼 주문 후보'를 만든다. 주문을 실제로 보내지 않는다."""
    r = df[df["date"] == pd.Timestamp(day)]
    if r.empty:
        return []
    r = r.iloc[0]
    checks = {
        "usable": bool(r["usable"]),
        "mis_ge_entry": bool(r["mis_pct"] >= p.entry_pct),
        "dte_ok": bool(r["dte"] >= p.min_dte),
        "no_dividend": bool(not (p.skip_dividend and r["div"] > 0)),
    }
    out = {"date": str(pd.Timestamp(day).date()), "und_code": und_code, "front_code": r["front_code"],
           "mis_pct": round(float(r["mis_pct"]), 3), "checks": checks, "params": asdict(p)}
    if all(checks.values()):
        out["action"] = "ENTER_CANDIDATE"
        out["legs"] = [
            {"instrument": und_code, "side": "BUY", "qty": p.qty, "ref_price": float(r["spot"])},
            {"instrument": r["front_code"], "side": "SELL", "qty": p.qty // 10, "ref_price": float(r["front_close"])},
        ]
    else:
        out["action"] = "NO_TRADE"
    return [out]
