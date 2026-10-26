"""08-3. 모델 학습과 진단: 연도별 성능, 기준선 비교, 보정, 피처 중요도, 시드 편차.

세 레이블을 각각 시드 5개로 워크포워드 학습(2022~2026)하고 예측을 저장한다(8-3, 8-4에서 사용).
변동성은 평균(회귀)과 함께 상단 80% 분위(분위 회귀)도 학습한다 — 불확실성을 반영한 보수적 예측.
선행: 01_targets_features.py
실행: uv run python examples/chapter08/03_model_diagnostics.py   (수 분 걸린다)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hedging.features.hedge_features import FEATURES
from hedging.models.lgbm_hedge import walk_forward, walk_forward_har

pd.set_option("display.width", 200)
panel = pd.read_parquet("data/processed/ch08_panel.parquet")
YEARS = [2022, 2023, 2024, 2025, 2026]
SEEDS = [1, 2, 3, 4, 5]


def r2(y, p):
    return 1 - ((y - p) ** 2).mean() / ((y - y.mean()) ** 2).mean()


preds, imps = {}, []
for target in ["y_crash", "y_up", "y_logvol"]:
    p, models = walk_forward(panel, target, YEARS, FEATURES, SEEDS)
    preds[target] = p
    for y, s, m in models:
        g = pd.Series(m.feature_importance("gain"), index=FEATURES)
        imps.append(pd.DataFrame({"target": target, "year": y, "seed": s, "feature": FEATURES, "gain": g / g.sum()}))
    p[["code", "date", "pred", "pred_std"] + [f"pred_s{s}" for s in SEEDS]].to_parquet(
        f"data/processed/ch08_pred_{target}.parquet", index=False)

# 변동성 기준선: HAR(선형) 과 '오늘 20일 변동성이 그대로 간다'(학습 없음)
# 상단 80% 분위 변동성 (8-3: 불확실성 반영)
q, _ = walk_forward(panel, "y_logvol", YEARS, FEATURES, SEEDS, params={"objective": "quantile", "alpha": 0.8})
q[["code", "date", "pred", "pred_std"] + [f"pred_s{s}" for s in SEEDS]].to_parquet(
    "data/processed/ch08_pred_y_logvol_q80.parquet", index=False)

har = walk_forward_har(panel, YEARS)
panel.assign(pred=har).dropna(subset=["pred"])[["code", "date", "pred"]].to_parquet(
    "data/processed/ch08_pred_har.parquet", index=False)
v = preds["y_logvol"].merge(panel.assign(har=har)[["code", "date", "har"]], on=["code", "date"], how="left")
v["naive"] = np.log(v["vol_20"])

print("[연도별 성능 — 시드 5개 평균 예측]")
rows = []
for y in YEARS:
    c = preds["y_crash"].query("test_year == @y").dropna(subset=["y_crash"])
    u = preds["y_up"].query("test_year == @y").dropna(subset=["y_up"])
    w = v.query("test_year == @y").dropna(subset=["y_logvol", "har"])
    rows.append({"연도": y, "급락 AUC": round(roc_auc_score(c["y_crash"], c["pred"]), 3),
                 "상승 AUC": round(roc_auc_score(u["y_up"], u["pred"]), 3),
                 "변동성 R² LGBM": round(r2(w["y_logvol"], w["pred"]), 3),
                 "변동성 R² HAR": round(r2(w["y_logvol"], w["har"]), 3),
                 "변동성 R² 그대로": round(r2(w["y_logvol"], w["naive"]), 3),
                 "학습 행": int(c["n_train"].iloc[0])})
c = preds["y_crash"].dropna(subset=["y_crash"])
u = preds["y_up"].dropna(subset=["y_up"])
w = v.dropna(subset=["y_logvol", "har"])
rows.append({"연도": "전체", "급락 AUC": round(roc_auc_score(c["y_crash"], c["pred"]), 3),
             "상승 AUC": round(roc_auc_score(u["y_up"], u["pred"]), 3),
             "변동성 R² LGBM": round(r2(w["y_logvol"], w["pred"]), 3),
             "변동성 R² HAR": round(r2(w["y_logvol"], w["har"]), 3),
             "변동성 R² 그대로": round(r2(w["y_logvol"], w["naive"]), 3), "학습 행": ""})
print(pd.DataFrame(rows).to_string(index=False))
qq = q.dropna(subset=["y_logvol"])
print(f"상단 80% 분위 예측: 실제 미래 변동성이 예측 이하인 비율 {(qq['y_logvol'] <= qq['pred']).mean():.3f} (목표 0.8)")

print("\n[시드 편차 — 시드별 점수의 범위]")
rows = []
for target in ["y_crash", "y_up", "y_logvol"]:
    d = preds[target].dropna(subset=[target])
    f = (lambda col: roc_auc_score(d[target], d[col])) if target != "y_logvol" else (lambda col: r2(d[target], d[col]))
    sc = [f(f"pred_s{s}") for s in SEEDS]
    rows.append({"레이블": target, "지표": "R²" if target == "y_logvol" else "AUC", "평균 예측": round(f("pred"), 3),
                 "시드 최저": round(min(sc), 3), "시드 최고": round(max(sc), 3),
                 "예측값 시드 표준편차(중앙값)": round(d["pred_std"].median(), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n[급락 확률 보정 — 예측 10분위별 실제 급락 비율]")
c = preds["y_crash"].dropna(subset=["y_crash"]).copy()
c["bin"] = pd.qcut(c["pred"], 10, labels=False) + 1
cal = c.groupby("bin").agg(예측평균=("pred", "mean"), 실제비율=("y_crash", "mean"), 행=("y_crash", "size"))
print(cal.round(3).T.to_string())

print("\n[변동성 예측 보정 — 예측 5분위별 실제 미래 변동성 중앙값%]")
w = v.dropna(subset=["y_logvol"]).copy()
w["bin"] = pd.qcut(w["pred"], 5, labels=False) + 1
print(w.groupby("bin").agg(예측=("pred", lambda s: np.exp(s).median() * 100),
                           실제=("fwd_vol", lambda s: s.median() * 100)).round(1).T.to_string())

print("\n[피처 중요도 — gain 비중, 연도·시드 평균 상위 8]")
imp = pd.concat(imps).groupby(["target", "feature"])["gain"].mean()
rows = []
for target in ["y_crash", "y_up", "y_logvol"]:
    top = imp[target].sort_values(ascending=False).head(8)
    rows.append({"레이블": target, **{str(i + 1): f"{k} {g:.0%}" for i, (k, g) in enumerate(top.items())}})
print(pd.DataFrame(rows).to_string(index=False))
