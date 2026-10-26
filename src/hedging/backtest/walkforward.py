"""워크포워드 검증 (6장).

후보 규칙(진입 기준 × 청산 방식)을 미리 정해 두고,
- 학습 구간: 시험 연도 이전의 모든 거래(확장 창)로 규칙별 평균 순손익(bp)을 계산
- 선택: 거래 수 min_trades 이상이면서 평균이 양수인 규칙 중 최고. 없으면 '무거래'
- 시험: 선택한 규칙의 시험 연도 거래만 성과로 집계
규칙 자체에는 학습되는 숫자가 없으므로, 거래 목록을 한 번 만들어 두고 구간별로 잘라 쓴다.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from hedging.strategies.basis_rules import BasisRuleParams, run

NO_EARLY_EXIT = -1e9  # 조기 청산 없이 만기까지 보유


def candidate_rules(base: BasisRuleParams) -> dict[str, BasisRuleParams]:
    rules = {}
    for th in [0.3, 0.5, 0.75, 1.0]:
        rules[f"진입{th}%·수렴청산"] = replace(base, entry_pct=th, exit_pct=0.0)
        rules[f"진입{th}%·만기보유"] = replace(base, entry_pct=th, exit_pct=NO_EARLY_EXIT)
    return rules


def pool_trades(features: dict[str, pd.DataFrame], rules: dict[str, BasisRuleParams]) -> pd.DataFrame:
    out = []
    for code, df in features.items():
        for name, p in rules.items():
            t = run(df, p)
            if len(t):
                t.insert(0, "rule", name)
                t.insert(1, "und_code", code)
                out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def walk_forward(pool: pd.DataFrame, test_years: list[int], min_trades: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, oos = [], []
    year = pool["entry_date"].dt.year
    for y in test_years:
        train = pool[year < y]
        stats = train.groupby("rule")["net_bp"].agg(["size", "mean"])
        ok = stats[(stats["size"] >= min_trades) & (stats["mean"] > 0)]
        chosen = ok["mean"].idxmax() if len(ok) else "무거래"
        test = pool[(year == y) & (pool["rule"] == chosen)] if chosen != "무거래" else pool.iloc[0:0]
        best_train = stats["mean"].idxmax() if len(stats) else None
        rows.append({
            "시험연도": y, "학습 거래(전 규칙)": len(train), "선택 규칙": chosen,
            "학습 평균bp": round(float(ok.loc[chosen, "mean"]), 1) if chosen != "무거래" else np.nan,
            "학습 최고 규칙": best_train,
            "학습 최고bp": round(float(stats["mean"].max()), 1) if len(stats) else np.nan,
            "시험 거래": len(test),
            "시험 평균bp": round(float(test["net_bp"].mean()), 1) if len(test) else np.nan,
        })
        oos.append(test.assign(test_year=y))
    return pd.DataFrame(rows), pd.concat(oos, ignore_index=True)
