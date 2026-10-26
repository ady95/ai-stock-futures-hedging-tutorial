"""06-2. 증거금·강제청산과 레그 체결 위험.

실행: uv run python examples/chapter06/03_margin_and_legging.py
"""

from dataclasses import replace

import pandas as pd

from hedging.backtest.ledger import LedgerConfig, run_ledger
from hedging.backtest.report import legging_mc
from hedging.strategies.basis_rules import BasisRuleParams, run

pd.set_option("display.width", 160)
base = BasisRuleParams(entry_pct=0.3, exec="next_open")  # 거래를 늘려 증거금 사건을 더 많이 만나게

print("[1] 예탁 여유에 따른 추가증거금·강제청산 (진입 0.3%, 진입 1억)")
rows = []
for code in ["005930", "000660"]:
    df = pd.read_parquet(f"data/processed/universe/{code}_features.parquet")
    for label, cap in [("여유 없음(1.36억+100만)", 136_000_000 + 1_000_000), ("여유 20%(1.56억)", 156_000_000),
                       ("여유 64%(2억)", 200_000_000)]:
        led, tr = run_ledger(df, base, LedgerConfig(capital=cap, target_notional=100_000_000))
        forced = tr[tr["reason"] == "강제청산"]
        rows.append({"종목": code, "자본": label, "거래": len(tr), "추가증거금 일수": int(led["margin_call"].sum()),
                     "강제청산": len(forced),
                     "강제청산 평균bp": round(forced["net_bp"].mean(), 1) if len(forced) else None,
                     "전체 평균bp": round(tr["net_bp"].mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n[2] 강제청산 사례 (삼성전자, 여유 없음)")
df = pd.read_parquet("data/processed/universe/005930_features.parquet")
led, tr = run_ledger(df, base, LedgerConfig(capital=137_000_000, target_notional=100_000_000))
for _, t in tr[tr["reason"] == "강제청산"].head(3).iterrows():
    w = led[(led["date"] >= t["entry_date"]) & (led["date"] <= t["exit_date"])]
    s = df.set_index("date").loc[w["date"], ["spot", "front_settle"]]
    print(f"  {t['code']} {t['entry_date']:%Y-%m-%d} 진입 → {t['exit_date']:%Y-%m-%d} 강제청산, 순손익 {t['net']:,.0f}원 ({t['net_bp']:.1f}bp)")
    for d, r in s.iterrows():
        x = w.set_index("date").loc[d]
        print(f"    {d:%Y-%m-%d} 현물 {r['spot']:>9,.0f} 선물정산 {r['front_settle']:>9,.0f} "
              f"선물계좌 {x['fut_account']:>13,.0f} 현금 {x['cash']:>13,.0f} {'추가증거금' if x['margin_call'] else ''}")

print("\n[3] 레그 체결 위험 시뮬레이션 (진입 0.5%, 선물 레그가 확률 p로 하루 늦게 체결, 500회)")
rows = []
for code in ["005930", "000660"]:
    df = pd.read_parquet(f"data/processed/universe/{code}_features.parquet")
    trades = run(df, replace(base, entry_pct=0.5))
    for p in [0.0, 0.1, 0.3]:
        mc = legging_mc(trades, df, p, n_sims=500, seed=1)
        rows.append({"종목": code, "지연 확률": p, "거래": len(trades),
                     "평균bp(시뮬 평균)": round(mc["평균bp"].mean(), 1),
                     "평균bp 범위(5~95%)": f"{mc['평균bp'].quantile(0.05):.1f} ~ {mc['평균bp'].quantile(0.95):.1f}",
                     "거래 하위5%bp": round(mc["하위5%bp"].mean(), 1), "최악bp": round(mc["최악bp"].mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))
