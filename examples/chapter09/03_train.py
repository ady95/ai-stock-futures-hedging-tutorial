"""09-3. PPO 정책 워크포워드 학습 (9-4 끝의 SAC 강건성 점검도 같은 스크립트).

시험 연도 2022~2026 × 시드 5개 = 25개 모델. 시험 연도마다 그 이전 데이터(삼성전자·SK하이닉스)로만 학습한다.
여러 프로세스로 나눠 돌린다(CPU 코어 수에 맞춰 --workers 조정).
선행: examples/chapter08/01_targets_features.py (패널)
실행:
  uv run python examples/chapter09/03_train.py                    # PPO, 200,000스텝 (6프로세스 약 20분)
  uv run python examples/chapter09/03_train.py --algo sac --timesteps 100000
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from hedging.rl.train import TIMESTEPS, load_frames, train_one

YEARS = [2022, 2023, 2024, 2025, 2026]
SEEDS = [1, 2, 3, 4, 5]


def job(args):
    year, seed, algo, steps = args
    panel = pd.read_parquet("data/processed/ch08_panel.parquet")
    t = time.time()
    path = train_one(load_frames(panel), year, seed, algo, steps)
    return year, seed, path, time.time() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--timesteps", type=int, default=TIMESTEPS)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(y, s, a.algo, a.timesteps) for y in YEARS for s in SEEDS]
    with ProcessPoolExecutor(a.workers) as ex:
        for year, seed, path, sec in ex.map(job, jobs):
            print(f"{a.algo} {year} seed {seed}: {sec / 60:.1f}분 → {path}", flush=True)
    print(f"전체 {len(jobs)}개 모델, {(time.time() - t0) / 60:.1f}분")
