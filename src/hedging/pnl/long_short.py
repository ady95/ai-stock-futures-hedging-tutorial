"""동일 종목 롱·숏 손익 계산 (1장).

세 가지 방식을 비교한다.

- hold_both_pnl   : 같은 수량을 같은 날 진입하고 같은 날 청산
- split_exit_pnl  : 한쪽이 목표 수익에 닿으면 먼저 청산하고, 남은 쪽은 본전 복귀 또는 만기에 청산
- hindsight_pnl   : 경로를 다 본 뒤 최고가에 롱, 최저가에 숏을 청산 (실전에서 불가능한 착시)

숏 레그는 주식 차입 매도(대주)를 가정한다. 비용 숫자는 예시이며 실제 요율은 증권사·연도별로 다르다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    fee_rate: float = 0.00015  # 매매수수료, 편도·체결금액 대비 (예시)
    sell_tax_rate: float = 0.0020  # 주식 매도 시 세금 (예시) — 공매도 매도에도 적용
    borrow_rate: float = 0.03  # 주식 차입 수수료, 연율 (예시)

    def long_cost(self, p_in: float, p_out: float, qty: float) -> float:
        return self.fee_rate * (p_in + p_out) * qty + self.sell_tax_rate * p_out * qty

    def short_cost(self, p_in: float, p_out: float, qty: float, days: int) -> float:
        borrow = self.borrow_rate * p_in * qty * days / 365
        return self.fee_rate * (p_in + p_out) * qty + self.sell_tax_rate * p_in * qty + borrow


ZERO_COST = CostModel(0.0, 0.0, 0.0)


@dataclass
class LongShortResult:
    pnl_long: float
    pnl_short: float
    cost: float
    exit_day_long: int
    exit_day_short: int
    first_leg: str | None = None  # 먼저 청산된 레그: "long" / "short" / None(동시 청산)
    recovered: bool = False  # 남은 레그가 본전 복귀로 청산되었는가 (분리 청산 전용)

    @property
    def gross(self) -> float:
        return self.pnl_long + self.pnl_short

    @property
    def net(self) -> float:
        return self.gross - self.cost

    @property
    def one_leg_days(self) -> int:
        """한쪽만 남아 방향 위험에 노출된 일수."""
        return abs(self.exit_day_long - self.exit_day_short)


def hold_both_pnl(
    s0: float, s1: float, qty: float, days: int, cost: CostModel = ZERO_COST
) -> LongShortResult:
    """같은 수량을 동시에 진입·청산했을 때의 손익."""
    return LongShortResult(
        pnl_long=(s1 - s0) * qty,
        pnl_short=(s0 - s1) * qty,
        cost=cost.long_cost(s0, s1, qty) + cost.short_cost(s0, s1, qty, days),
        exit_day_long=days,
        exit_day_short=days,
    )


def split_exit_pnl(
    path: np.ndarray, qty: float, take_profit: float = 0.05, cost: CostModel = ZERO_COST
) -> LongShortResult:
    """청산 시점 분리 전략.

    1) 진입일에 롱·숏을 같은 수량으로 보유한다.
    2) 가격이 +take_profit 이상이면 롱을, -take_profit 이하이면 숏을 먼저 청산한다.
    3) 남은 레그는 가격이 진입가로 돌아오면(본전) 청산하고, 끝까지 안 돌아오면 마지막 날 청산한다.
    가격은 일별 종가로만 관찰하므로 목표가를 넘어선 종가에 체결된다고 가정한다.
    """
    s0 = float(path[0])
    last = len(path) - 1
    exit_long = exit_short = last
    first_leg = None

    t = 1
    while t <= last:
        s = path[t]
        if s >= s0 * (1 + take_profit):
            first_leg, exit_long = "long", t
            break
        if s <= s0 * (1 - take_profit):
            first_leg, exit_short = "short", t
            break
        t += 1

    recovered = False
    if first_leg == "long":  # 남은 숏: 가격이 진입가 이하로 내려오면 청산
        for u in range(exit_long + 1, last + 1):
            if path[u] <= s0:
                exit_short, recovered = u, True
                break
    elif first_leg == "short":  # 남은 롱: 가격이 진입가 이상으로 오르면 청산
        for u in range(exit_short + 1, last + 1):
            if path[u] >= s0:
                exit_long, recovered = u, True
                break

    p_long, p_short = float(path[exit_long]), float(path[exit_short])
    return LongShortResult(
        pnl_long=(p_long - s0) * qty,
        pnl_short=(s0 - p_short) * qty,
        cost=cost.long_cost(s0, p_long, qty) + cost.short_cost(s0, p_short, qty, exit_short),
        exit_day_long=exit_long,
        exit_day_short=exit_short,
        first_leg=first_leg,
        recovered=recovered,
    )


def hindsight_pnl(path: np.ndarray, qty: float, cost: CostModel = ZERO_COST) -> LongShortResult:
    """사후 최적 청산: 롱은 최고가, 숏은 최저가에 청산. 미래를 알아야만 가능하다."""
    s0 = float(path[0])
    i_max, i_min = int(np.argmax(path)), int(np.argmin(path))
    p_long, p_short = float(path[i_max]), float(path[i_min])
    return LongShortResult(
        pnl_long=(p_long - s0) * qty,
        pnl_short=(s0 - p_short) * qty,
        cost=cost.long_cost(s0, p_long, qty) + cost.short_cost(s0, p_short, qty, i_min),
        exit_day_long=i_max,
        exit_day_short=i_min,
        first_leg="long" if i_max < i_min else ("short" if i_min < i_max else None),
    )
