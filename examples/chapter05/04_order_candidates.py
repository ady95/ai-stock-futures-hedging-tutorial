"""05-4. 규칙 기반 전략의 주문 후보 신호 생성 (주문을 보내지 않는다).

그날 종가로 규칙을 평가해 '다음 거래일에 낼 주문 후보'를 JSON Lines로 남긴다.
실행: uv run python examples/chapter05/04_order_candidates.py [기준일=데이터 마지막 날]
"""

import json
import sys
from pathlib import Path

import pandas as pd

from hedging.data.collect import load_futures, load_spot
from hedging.strategies.basis_rules import BasisRuleParams, attach_opens, order_candidates

p = BasisRuleParams(entry_pct=0.5, exec="next_open")
out = Path("outputs")
out.mkdir(exist_ok=True)
log = []
for und in ["005930", "000660"]:
    fv = pd.read_parquet(f"data/processed/{und}_features.parquet")
    df = attach_opens(fv, load_futures(und), load_spot(und))
    day = sys.argv[1] if len(sys.argv) > 1 else df["date"].max()
    cands = order_candidates(df, p, und, day)
    log += cands
    for c in cands:
        print(json.dumps({k: v for k, v in c.items() if k != "params"}, ensure_ascii=False, indent=1))
    # 과거 신호 이력: 규칙을 모두 통과한 날
    hist = [c for d in df["date"] for c in order_candidates(df, p, und, d) if c["action"] == "ENTER_CANDIDATE"]
    print(f"  {und} 과거 진입 후보일 {len(hist)}일, 최근 3건: {[h['date'] for h in hist[-3:]]}")

path = out / "ch05_order_candidates.jsonl"
with path.open("w", encoding="utf-8") as f:
    for c in log:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(f"저장: {path}")
