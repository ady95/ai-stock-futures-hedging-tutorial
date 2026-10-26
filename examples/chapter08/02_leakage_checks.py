"""08-2. 시점 누수 방지: 같은 모델을 잘못된 방법으로 검증하면 얼마나 좋아 보이나.

비교
 A. 올바른 방법: 연 단위 워크포워드 + 5거래일 엠바고
 B. 무작위 5겹 교차검증: 날짜를 섞어 나눈다 → 겹치는 레이블(미래 5일)을 통해 정답이 샌다
 C. 미래가 섞인 피처: 중심 이동창 변동성(앞뒤 10일) 하나를 피처에 추가 → 워크포워드라도 누수
그리고 피처 계산 자체가 미래를 보지 않는지 '자르기 테스트'로 확인하고,
시험 구간의 피처가 학습 범위를 벗어나는지(범위 이탈) 레이블 없이 점검한다.

선행: 01_targets_features.py
실행: uv run python examples/chapter08/02_leakage_checks.py
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from hedging.backtest.hedge_ledger import load_hedge_frame
from hedging.features.hedge_features import FEATURES, build_panel
from hedging.models.lgbm_hedge import BASE_PARAMS, range_shift, walk_forward

panel = pd.read_parquet("data/processed/ch08_panel.parquet")
YEARS = [2022, 2023, 2024, 2025, 2026]


def r2(y, p):
    return 1 - ((y - p) ** 2).mean() / ((y - y.mean()) ** 2).mean()


def score(target, pred_df):
    d = pred_df.dropna(subset=[target])
    return roc_auc_score(d[target], d["pred"]) if target == "y_crash" else r2(d[target], d["pred"])


rows = []
for target in ["y_crash", "y_logvol"]:
    # A. 워크포워드
    a, _ = walk_forward(panel, target, YEARS, FEATURES, seeds=[1])
    # B. 무작위 5겹 (같은 행 수, 같은 하이퍼파라미터, 고정 반복 수)
    d = panel.dropna(subset=FEATURES + [target]).reset_index(drop=True)
    obj = "binary" if target == "y_crash" else "regression"
    pred = np.zeros(len(d))
    for tr, te in KFold(5, shuffle=True, random_state=1).split(d):
        m = lgb.train({**BASE_PARAMS, "objective": obj, "seed": 1}, lgb.Dataset(d.loc[tr, FEATURES], d.loc[tr, target]),
                      num_boost_round=300)
        pred[te] = m.predict(d.loc[te, FEATURES])
    b = d.assign(pred=pred)
    # C. 미래가 섞인 피처 하나 추가
    p2 = panel.copy()
    p2["vol_center"] = p2.groupby("code")["lr"].transform(lambda s: s.rolling(21, center=True, min_periods=21).std())
    c, _ = walk_forward(p2, target, YEARS, FEATURES + ["vol_center"], seeds=[1])
    metric = "AUC" if target == "y_crash" else "R²"
    rows.append({"레이블": target, "지표": metric, "A 워크포워드(올바름)": round(score(target, a), 3),
                 "B 무작위 5겹": round(score(target, b), 3), "C 미래 섞인 피처": round(score(target, c), 3)})
print("[검증 방법별 성능 — 같은 모델, 같은 데이터]")
print(pd.DataFrame(rows).to_string(index=False))

# 자르기 테스트: 데이터 뒤쪽을 잘라도 남은 날의 피처가 그대로여야 한다(시장 피처 포함, 패널 단위)
frames = {c: load_hedge_frame(c) for c in ["005930", "000660", "005380"]}
full = build_panel(frames)
cut_date = full["date"].sort_values().unique()[1200]
cut = build_panel({c: f[f["date"] < cut_date] for c, f in frames.items()})
merged = cut.merge(full, on=["code", "date"], suffixes=("_cut", "_full"))
same = all(np.allclose(merged[f"{k}_cut"], merged[f"{k}_full"], equal_nan=True) for k in FEATURES)
last_days = cut.sort_values("date").groupby("code").tail(5)
print(f"\n[자르기 테스트] {cut_date:%Y-%m-%d} 이전 {len(merged):,}행의 피처 {len(FEATURES)}개 일치: {same}, "
      f"잘린 끝 5일의 레이블은 비어 있음: {bool(last_days['y_crash'].isna().all())}")

# 범위 이탈 점검: 시험 연도 피처가 학습 구간 범위 밖인 비율 (레이블을 보지 않으므로 사전에 할 수 있다)
shift = range_shift(panel, FEATURES + ["rate"], YEARS)
top = shift.max().sort_values(ascending=False).head(6).index
print("\n[범위 이탈 — 시험 행 중 학습 범위 밖 비율, 최대인 피처 6개]")
print((shift[top] * 100).round(1).to_string())

# rate를 피처에 넣었을 때와 뺐을 때 (8-2 본문: 이 비교는 시험 결과를 본 것이므로 공개해 둔다)
rows = []
for name, feats in [("rate 포함", FEATURES + ["rate"]), ("rate 제외(채택)", FEATURES)]:
    c, _ = walk_forward(panel, "y_crash", YEARS, feats, seeds=[1])
    v, _ = walk_forward(panel, "y_logvol", YEARS, feats, seeds=[1])
    rows.append({"피처": name, "급락 AUC": round(score("y_crash", c), 3), "변동성 R²": round(score("y_logvol", v), 3),
                 "2022 변동성 R²": round(score("y_logvol", v[v["test_year"] == 2022]), 3)})
print("\n[금리 수준 피처의 영향]")
print(pd.DataFrame(rows).to_string(index=False))
