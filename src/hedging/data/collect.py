"""월물별 주식선물 일봉 + 기초주식 일봉 수집 → Parquet 저장 (3장).

저장 구조 (data/raw 는 커밋하지 않는다)
  data/raw/kis/futures_daily/<기초주식코드>/<단축코드>.parquet
  data/raw/kis/stock_daily/<기초주식코드>.parquet

만기가 지난 월물 파일은 다시 받지 않는다(캐시). 진행 중인 월물과 현물은 매번 갱신한다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from hedging.data.codes import futures_code, months_between
from hedging.data.kis import KisClient, yyyymmdd
from hedging.instruments.contracts import last_trading_day

RAW = Path("data/raw/kis")


def collect_underlying(
    client: KisClient,
    und_code: str,
    und_id: str,
    start_ym: tuple[int, int],
    end_ym: tuple[int, int],
    today: date | None = None,
    root: Path = RAW,
) -> pd.DataFrame:
    """기초자산 하나의 모든 결제월 일봉을 받는다. 결제월마다 행 수를 요약해 돌려준다."""
    today = today or date.today()
    out_dir = root / "futures_daily" / und_code
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for y, m in months_between(start_ym, end_ym):
        code = futures_code(und_id, y, m)
        path = out_dir / f"{code}.parquet"
        expired = last_trading_day(y, m) < today
        if expired and path.exists():
            df = pd.read_parquet(path)
            status = "cache"
        else:
            # 옛 체계 시기에는 12월물이 약 3년 전부터 상장되어 있었다(실측). 넉넉히 4년 전부터 요청한다
            df = client.stock_futures_daily(code, f"{y - 4}0101", yyyymmdd(min(today, date(y, m, 28))))
            status = "fetch"
            if not df.empty:
                df.insert(0, "code", code)
                df.insert(1, "expiry_ym", f"{y}{m:02d}")
                df.to_parquet(path, index=False)
        summary.append({"expiry_ym": f"{y}{m:02d}", "code": code, "rows": len(df), "status": status,
                        "first": df["date"].min() if len(df) else None,
                        "last": df["date"].max() if len(df) else None})
    return pd.DataFrame(summary)


def collect_spot(client: KisClient, und_code: str, start: str, end: str, root: Path = RAW) -> pd.DataFrame:
    out_dir = root / "stock_daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = client.stock_daily(und_code, start, end, adjusted=False)
    df.to_parquet(out_dir / f"{und_code}.parquet", index=False)
    return df


def collect_dividends(client: KisClient, und_code: str, start: str, end: str, root: Path = RAW) -> pd.DataFrame:
    out_dir = root / "dividends"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = client.dividend_schedule(und_code, start, end)
    df.to_parquet(out_dir / f"{und_code}.parquet", index=False)
    return df


def load_dividends(und_code: str, root: Path = RAW) -> pd.DataFrame:
    return pd.read_parquet(root / "dividends" / f"{und_code}.parquet")


def load_futures(und_code: str, root: Path = RAW) -> pd.DataFrame:
    files = sorted((root / "futures_daily" / und_code).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{und_code}: 수집된 선물 일봉이 없습니다 (03_collect_daily.py 먼저 실행)")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_spot(und_code: str, root: Path = RAW) -> pd.DataFrame:
    return pd.read_parquet(root / "stock_daily" / f"{und_code}.parquet")
