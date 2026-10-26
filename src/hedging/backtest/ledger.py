"""현물·선물 통합 장부 백테스터 (6장).

5장 규칙(basis_rules)과 같은 진입·청산 판단을 하되, 하루 단위로 현금 흐름을 장부에 남긴다.

- 현금(cash): 기준금리로 이자를 받는다 → '무거래 = 현금 보유' 기준선과 공정하게 비교
- 현물(spot_qty): 종가로 평가
- 선물 계좌(fut_account): 진입 때 위탁증거금을 현금에서 옮기고, 매일 KRX 정산가로 일일정산
  유지증거금 아래면 현금에서 위탁증거금 수준까지 채운다(추가증거금). 현금이 모자라면 다음 날 시가에 강제청산
- 만기: 선물은 최종결제가격(= 현물 종가)으로 정산, 현물은 종가에 매도
- 체결: next_open(기본) — 신호 다음 거래일 시가 / close — 신호일 종가

수수료·세금·슬리피지 가정은 BasisRuleParams 를 그대로 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hedging.instruments.contracts import tick_size as futures_tick
from hedging.instruments.stock_ticks import stock_tick
from hedging.strategies.basis_rules import BasisRuleParams


@dataclass(frozen=True)
class LedgerConfig:
    capital: float = 200_000_000  # 시작 자본(원)
    target_notional: float = 100_000_000  # 진입 시 현물 명목금액 목표(원)
    fixed_qty: int | None = None  # 지정하면 target_notional 대신 고정 수량(5장과 비교용)
    maint_rate: float = 0.24  # 유지증거금률


@dataclass
class _Pos:
    code: str
    ltd: pd.Timestamp
    qty: int
    entry_date: pd.Timestamp
    entry_spot: float
    entry_fut: float
    last_settle: float
    costs: float = 0.0
    events: list = field(default_factory=list)


def _signal(r: pd.Series, p: BasisRuleParams) -> bool:
    return bool(r["usable"] and r["mis_pct"] >= p.entry_pct and r["dte"] >= p.min_dte
                and not (p.skip_dividend and r["div"] > 0) and r["front_volume"] >= p.min_volume)


def run_ledger(df: pd.DataFrame, p: BasisRuleParams, cfg: LedgerConfig = LedgerConfig()
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """일별 장부와 거래 목록을 돌려준다."""
    rows = df.reset_index(drop=True)
    cash, fut_acct, spot_qty = cfg.capital, 0.0, 0
    pos: _Pos | None = None
    pending: tuple[str, str] | None = None  # (행동, 사유) — 다음 거래일 시가에 실행
    trades, book = [], []
    prev_date = None
    cum = {"spot_pnl": 0.0, "fut_pnl": 0.0, "costs": 0.0, "interest": 0.0}

    def costs_spot(px, qty, sell, day):
        c = p.spot_fee * px * qty + p.slip_ticks * stock_tick(px, day) * qty
        return c + (p.sell_tax * px * qty if sell else 0.0)

    def costs_fut(px, qty, settle=False):
        return p.fut_fee * px * qty + (0.0 if settle else p.slip_ticks * futures_tick(px) * qty)

    def open_position(day, s_px, f_px, r):
        nonlocal cash, fut_acct, spot_qty, pos
        qty = cfg.fixed_qty or int(cfg.target_notional // (s_px * 10)) * 10
        # 현금으로 감당할 수 있는 만큼만: 현물 대금 + 위탁증거금 + 비용(넉넉히 1%)
        per_share = s_px + f_px * p.margin_rate + s_px * 0.01
        qty = min(qty, int(cash // (per_share * 10)) * 10)
        if qty <= 0:
            return
        c = costs_spot(s_px, qty, False, day) + costs_fut(f_px, qty)
        margin = f_px * qty * p.margin_rate
        cash -= s_px * qty + c + margin
        fut_acct += margin
        spot_qty = qty
        cum["costs"] += c
        pos = _Pos(r["front_code"], r["ltd"], qty, day, s_px, f_px, f_px, c)

    def close_position(day, s_px, f_px, reason, final=False):
        nonlocal cash, fut_acct, spot_qty, pos
        q = pos.qty
        var = (pos.last_settle - f_px) * q  # 매도 포지션: 가격 하락이 이익
        fut_acct += var
        cum["fut_pnl"] += var
        c = costs_spot(s_px, q, True, day) + costs_fut(f_px, q, settle=final)
        cash += s_px * q - c + fut_acct
        cum["spot_pnl"] += (s_px - pos.entry_spot) * q
        cum["costs"] += c
        trades.append({"code": pos.code, "entry_date": pos.entry_date, "exit_date": day, "reason": reason,
                       "qty": q, "days": (day - pos.entry_date).days,
                       "spot_pnl": (s_px - pos.entry_spot) * q, "fut_pnl": (pos.entry_fut - f_px) * q,
                       "costs": pos.costs + c, "notional": pos.entry_spot * q, "events": ",".join(pos.events)})
        fut_acct, spot_qty, pos = 0.0, 0, None

    for _, r in rows.iterrows():
        d = r["date"]
        # 1) 현금 이자
        if prev_date is not None:
            it = cash * r["rate"] * (d - prev_date).days / 365
            cash += it
            cum["interest"] += it
        prev_date = d
        call = False
        # 2) 전날 결정한 주문을 오늘 시가에 실행
        if pending is not None:
            act, why = pending
            pending = None
            if act == "enter" and pos is None and r["front_code"] == r["front_code"]:
                s_px, f_px = r["spot_open"], r["front_open"]
                if not (pd.isna(s_px) or pd.isna(f_px)):
                    open_position(d, s_px, f_px, r)
            elif act == "exit" and pos is not None:
                f_px = r["front_open"] if r["front_code"] == pos.code else np.nan
                if not (pd.isna(r["spot_open"]) or pd.isna(f_px)):
                    if why == "강제청산":
                        pos.events.append("강제청산")
                    close_position(d, r["spot_open"], f_px, why)
        # 3) 만기 또는 일일정산
        if pos is not None and d >= pos.ltd:
            close_position(d, r["spot"], r["spot"], "만기", final=True)
        elif pos is not None:
            settle = r["front_settle"] if not pd.isna(r.get("front_settle", np.nan)) else r["front_close"]
            var = (pos.last_settle - settle) * pos.qty
            fut_acct += var
            cum["fut_pnl"] += var
            pos.last_settle = settle
            maint = settle * pos.qty * cfg.maint_rate
            if fut_acct < maint:
                need = settle * pos.qty * p.margin_rate - fut_acct
                call = True
                if cash >= need:
                    cash -= need
                    fut_acct += need
                    pos.events.append("추가증거금")
                else:
                    pending = ("exit", "강제청산")
        # 4) 오늘 종가로 다음 행동 결정
        if pending is None:
            if pos is None and _signal(r, p):
                if p.exec == "close":
                    open_position(d, r["spot"], r["front_close"], r)
                    if pos is not None:
                        pos.last_settle = r["front_settle"] if not pd.isna(r.get("front_settle", np.nan)) \
                            else r["front_close"]
                        # 종가 진입 직후 정산가와의 차이 반영
                        var = (r["front_close"] - pos.last_settle) * pos.qty
                        fut_acct += var
                        cum["fut_pnl"] += var
                else:
                    pending = ("enter", "신호")
            elif pos is not None and r["usable"] and r["mis_pct"] <= p.exit_pct and d > pos.entry_date \
                    and d < pos.ltd:
                if p.exec == "close":
                    close_position(d, r["spot"], r["front_close"], "수렴")
                else:
                    pending = ("exit", "수렴")
        spot_val = spot_qty * r["spot"]
        book.append({"date": d, "cash": cash, "fut_account": fut_acct, "spot_value": spot_val,
                     "equity": cash + fut_acct + spot_val, "in_position": pos is not None,
                     "margin_call": call, **cum})
    led = pd.DataFrame(book)
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["gross"] = tr["spot_pnl"] + tr["fut_pnl"]
        tr["net"] = tr["gross"] - tr["costs"]
        tr["net_bp"] = tr["net"] / tr["notional"] * 1e4
    return led, tr


def cash_benchmark(ledger: pd.DataFrame, df: pd.DataFrame, capital: float) -> pd.Series:
    """같은 자본을 현금으로만 들고 기준금리 이자를 받았을 때의 평가액(무거래 기준선)."""
    r = df.set_index("date")["rate"].reindex(ledger["date"]).values
    days = ledger["date"].diff().dt.days.fillna(0).values
    growth = np.cumprod(1 + r * days / 365)
    return pd.Series(capital * growth, index=ledger["date"])
