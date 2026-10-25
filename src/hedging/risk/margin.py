"""주식선물 증거금과 일일정산 (2장).

용어
- 위탁증거금: 신규 주문 시 투자자가 증권사에 맡겨야 하는 금액 (개시 수준)
- 유지증거금: 예탁금이 이 아래로 내려가면 추가증거금(마진콜)이 발생하는 기준
- 일일정산: 매일 정산가격으로 손익을 현금 정산한다. 미실현 손실도 그날 예탁금에서 빠진다

증거금률은 한국거래소가 매월 조정한다. 기본값은 2026년 5월 삼성전자 선물 예시
(거래 24.0% / 위탁 36.0% / 유지 24.0%)이며, 실제 값은 최신 공지로 바꿔 넣는다.
추가증거금은 '위탁증거금 수준까지' 채운다고 단순화한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hedging.instruments.contracts import MULTIPLIER


@dataclass(frozen=True)
class MarginRates:
    initial: float = 0.36  # 위탁증거금률
    maintenance: float = 0.24  # 유지증거금률


def notional(price: float, contracts: int, multiplier: float = MULTIPLIER) -> float:
    return price * contracts * multiplier


def initial_margin(price: float, contracts: int, rates: MarginRates, multiplier: float = MULTIPLIER) -> float:
    return notional(price, contracts, multiplier) * rates.initial


def maintenance_margin(price: float, contracts: int, rates: MarginRates, multiplier: float = MULTIPLIER) -> float:
    return notional(price, contracts, multiplier) * rates.maintenance


def margin_call_price(
    f0: float,
    contracts: int,
    rates: MarginRates,
    deposit: float | None = None,
    side: str = "short",
    multiplier: float = MULTIPLIER,
) -> float:
    """예탁금이 유지증거금 아래로 떨어지기 시작하는 선물가격.

    매도(short): 예탁금 - (F - F0)·N·M < F·N·M·m  →  F > (D + F0·N·M) / (N·M·(1 + m))
    매수(long) : 예탁금 + (F - F0)·N·M < F·N·M·m  →  F < (F0·N·M - D) / (N·M·(1 - m))
    """
    nm = contracts * multiplier
    d = initial_margin(f0, contracts, rates, multiplier) if deposit is None else deposit
    m = rates.maintenance
    if side == "short":
        return (d + f0 * nm) / (nm * (1 + m))
    return (f0 * nm - d) / (nm * (1 - m))


def daily_settlement(
    settle_prices: list[float],
    contracts: int,
    rates: MarginRates,
    side: str = "short",
    deposit: float | None = None,
    multiplier: float = MULTIPLIER,
) -> pd.DataFrame:
    """정산가격 경로에 따른 일일정산·추가증거금 시뮬레이션.

    settle_prices[0]은 진입가격(첫날 정산 전)이다.
    매일: 정산차금 반영 → 유지증거금 미달이면 위탁증거금 수준까지 추가 입금했다고 가정.
    """
    sign = -1 if side == "short" else 1
    f_prev = settle_prices[0]
    equity = initial_margin(f_prev, contracts, rates, multiplier) if deposit is None else deposit
    rows = [{"day": 0, "price": f_prev, "variation": 0.0, "equity": equity,
             "maint_req": maintenance_margin(f_prev, contracts, rates, multiplier),
             "call": False, "top_up": 0.0, "cum_top_up": 0.0}]
    cum = 0.0
    for day, f in enumerate(settle_prices[1:], start=1):
        variation = sign * (f - f_prev) * contracts * multiplier
        equity += variation
        maint = maintenance_margin(f, contracts, rates, multiplier)
        call = equity < maint
        top_up = initial_margin(f, contracts, rates, multiplier) - equity if call else 0.0
        equity += top_up
        cum += top_up
        rows.append({"day": day, "price": f, "variation": variation, "equity": equity,
                     "maint_req": maint, "call": call, "top_up": top_up, "cum_top_up": cum})
        f_prev = f
    return pd.DataFrame(rows)
