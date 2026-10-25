"""현물 수량 → 주식선물 계약 수 변환 (2장).

N = round(Q_spot × h / M)   (주식 수 기준)
N = round(Q_spot × S × h / (M × F))   (금액 기준)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hedging.instruments.contracts import MULTIPLIER


@dataclass(frozen=True)
class HedgePlan:
    spot_qty: int
    target_ratio: float
    multiplier: float
    contracts: int  # 매도할 선물 계약 수
    hedged_shares: float  # 계약 수 × 승수 (주식 환산 수량)

    @property
    def net_shares(self) -> float:
        """헤지 후 남는 순노출(주). 양수 = 순매수, 음수 = 순매도."""
        return self.spot_qty - self.hedged_shares

    @property
    def achieved_ratio(self) -> float:
        """정수 계약으로 실제 달성한 헤지 비율(주식 수 기준)."""
        return self.hedged_shares / self.spot_qty if self.spot_qty else 0.0


def _round(x: float, mode: str) -> int:
    if mode == "nearest":
        return int(math.floor(x + 0.5))  # 0.5는 올림 (파이썬 round의 은행가 반올림을 피함)
    if mode == "down":
        return int(math.floor(x))
    if mode == "up":
        return int(math.ceil(x))
    raise ValueError(f"rounding: {mode}")


def plan_hedge(
    spot_qty: int,
    hedge_ratio: float = 1.0,
    multiplier: float = MULTIPLIER,
    rounding: str = "nearest",
    spot_price: float | None = None,
    futures_price: float | None = None,
) -> HedgePlan:
    """헤지 계약 수를 계산한다.

    spot_price와 futures_price를 모두 주면 금액 기준(현물 평가액 = 선물 명목금액)으로,
    아니면 주식 수 기준으로 계산한다.
    """
    if spot_price and futures_price:
        raw = spot_qty * spot_price * hedge_ratio / (multiplier * futures_price)
    else:
        raw = spot_qty * hedge_ratio / multiplier
    n = _round(raw, rounding)
    return HedgePlan(spot_qty, hedge_ratio, multiplier, n, n * multiplier)


def exposure_krw(plan: HedgePlan, spot_price: float, futures_price: float) -> dict[str, float]:
    """현물 평가액, 선물 명목금액(매도), 순노출 금액."""
    spot_value = plan.spot_qty * spot_price
    fut_notional = plan.contracts * plan.multiplier * futures_price
    return {
        "spot_value": spot_value,
        "futures_notional": fut_notional,
        "net_value": spot_value - fut_notional,
    }
