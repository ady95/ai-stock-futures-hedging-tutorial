"""05-3. 체결 가정·비용·유동성이 결과를 어떻게 바꾸는가.

실행: uv run python examples/chapter05/03_costs_execution.py
"""

from dataclasses import replace

import pandas as pd

from hedging.data.collect import load_futures, load_spot
from hedging.data.krx import load_krx
from hedging.strategies.basis_rules import BasisRuleParams, attach_opens, decompose, run, summarize

pd.set_option("display.width", 200)
base = BasisRuleParams()
data = {}
for und in ["005930", "000660"]:
    fv = pd.read_parquet(f"data/processed/{und}_features.parquet")
    data[und] = attach_opens(fv, load_futures(und), load_spot(und))

print("[1] 진입 기준 × 체결 가정 (거래당 순손익 bp, 승률)")
rows = []
for und, df in data.items():
    for ex in ["close", "next_open"]:
        for th in [0.3, 0.5, 0.75, 1.0]:
            s = summarize(run(df, replace(base, entry_pct=th, exec=ex)))
            rows.append({"종목": und, "체결": ex, "진입%": th, "거래": s["trades"],
                         "승률": round(s["win_rate"], 2), "거래당bp": round(s["avg_net_bp"], 1)})
print(pd.DataFrame(rows).pivot_table(index=["종목", "진입%"], columns="체결", values=["거래", "승률", "거래당bp"])
      .to_string())

print("\n[2] 손익 분해 (진입 0.5%, 거래당 평균)")
dec = []
for und, df in data.items():
    for ex in ["close", "next_open"]:
        d = decompose(run(df, replace(base, exec=ex)))
        dec.append({"종목": und, "체결": ex, **{k: round(v, 3) for k, v in d.items()}})
print(pd.DataFrame(dec).to_string(index=False))

print("\n[3] 비용 가정 민감도 (삼성전자, 진입 0.5%, 거래당 순손익 bp)")
sens = []
for ex in ["close", "next_open"]:
    for slip in [0.0, 0.5, 1.0]:
        for tax in [0.0015, 0.0020, 0.0025]:
            s = summarize(run(data["005930"], replace(base, exec=ex, slip_ticks=slip, sell_tax=tax)))
            sens.append({"체결": ex, "슬리피지(틱)": slip, "매도세%": tax * 100, "거래당bp": round(s["avg_net_bp"], 1)})
print(pd.DataFrame(sens).pivot_table(index=["체결", "슬리피지(틱)"], columns="매도세%", values="거래당bp").to_string())

print("\n[4] 배당 구간 진입 허용 시 (삼성전자, 다음 날 시가)")
for skip in [True, False]:
    t = run(data["005930"], replace(base, skip_dividend=skip))
    s = summarize(t)
    print(f"  배당 구간 {'제외' if skip else '허용'}: 거래 {s['trades']}, 거래당 {s['avg_net_bp']:.1f}bp")

print("\n[5] 신호일 유동성 (KRX 미결제약정·거래량, 진입 0.5%)")
krx = load_krx("kospi")
for und, name in [("005930", "삼성전자"), ("000660", "SK하이닉스")]:
    t = run(data[und], base)
    k = krx[krx["und_name"] == name][["date", "short_code", "volume", "open_interest"]]
    m = t.merge(k, left_on=["signal_date", "code"], right_on=["date", "short_code"], how="left")
    allf = data[und].merge(k, left_on=["date", "front_code"], right_on=["date", "short_code"], how="left")
    print(f"  {name}: 신호일 거래량 중앙값 {m['volume'].median():,.0f} / 전체 중앙값 {allf['volume'].median():,.0f}, "
          f"신호일 미결제 중앙값 {m['open_interest'].median():,.0f} / 전체 {allf['open_interest'].median():,.0f}")
