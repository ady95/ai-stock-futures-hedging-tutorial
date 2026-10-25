"""합성 가격 경로 생성기.

실데이터 없이 손익 로직을 검증하기 위한 도구다. 결과는 seed로 재현된다.
"""

from __future__ import annotations

import numpy as np


def gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    days: int,
    n_paths: int,
    seed: int | None = None,
    trading_days: int = 252,
) -> np.ndarray:
    """기하 브라운 운동(GBM) 일별 가격 경로를 만든다.

    반환: shape (n_paths, days + 1). 0열은 모두 s0.
    mu, sigma는 연율 기준이다.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / trading_days
    z = rng.standard_normal((n_paths, days))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)], axis=1)
    return s0 * np.exp(log_path)


def _linear(points: list[tuple[int, float]], s0: float) -> np.ndarray:
    """(일자, 수익률) 꼭짓점을 직선으로 이은 경로."""
    days = [p[0] for p in points]
    rets = [p[1] for p in points]
    t = np.arange(days[-1] + 1)
    return s0 * (1.0 + np.interp(t, days, rets))


def scenario_paths(s0: float = 100_000, days: int = 20) -> dict[str, np.ndarray]:
    """설명용 대표 경로 5종.

    - 상승 지속: 0% → +10%
    - 하락 지속: 0% → -10%
    - 상승 후 복귀: +8%까지 올랐다가 -2%로 마감
    - 하락 후 복귀: -8%까지 내렸다가 +2%로 마감
    - 횡보: ±2% 안에서 등락
    """
    h = days // 2
    q = days // 4
    return {
        "상승 지속": _linear([(0, 0.0), (days, 0.10)], s0),
        "하락 지속": _linear([(0, 0.0), (days, -0.10)], s0),
        "상승 후 복귀": _linear([(0, 0.0), (h, 0.08), (days, -0.02)], s0),
        "하락 후 복귀": _linear([(0, 0.0), (h, -0.08), (days, 0.02)], s0),
        "횡보": _linear([(0, 0.0), (q, 0.02), (h, -0.02), (3 * q, 0.02), (days, 0.0)], s0),
    }
