"""KIS 주식선물옵션 종목 마스터 파일(fo_stk_code_mts.mst) 다운로드·파싱 (2장).

API 키 없이 받을 수 있는 공개 파일이다. 한국투자증권 공식 저장소의
stocks_info/domestic_stock_future_code.py 와 같은 주소·형식을 사용한다.
원본 파일은 data/raw/ 에만 저장하고 저장소에 커밋하지 않는다.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

MASTER_URL = "https://new.real.download.dws.co.kr/common/master/fo_stk_code_mts.mst.zip"
COLUMNS = ["kind", "short_code", "std_code", "name", "atm", "strike", "month_seq", "und_code", "und_name"]
KIND_FUTURES = "1"  # 상품종류 1 = 주식선물 (2 = 스프레드, 그 밖은 옵션 등)


def download_master(raw_dir: str | Path = "data/raw") -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(MASTER_URL, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(raw_dir)
    return raw_dir / "fo_stk_code_mts.mst"


def load_stock_futures(mst_path: str | Path) -> pd.DataFrame:
    """주식선물 행만 골라 결제월·승수 열을 붙인다."""
    df = pd.read_table(mst_path, sep="|", encoding="cp949", header=None, dtype=str, names=COLUMNS)
    df = df.apply(lambda c: c.str.strip())
    fut = df[df["kind"] == KIND_FUTURES].copy()
    # 종목명 예: "삼성전자   F 202611 (  10)" → 결제월 202611, 승수 10
    fut["expiry_ym"] = fut["name"].str.extract(r"F (\d{6})")[0]
    fut["multiplier"] = pd.to_numeric(fut["name"].str.extract(r"\(\s*([\d.]+)\)")[0])
    return fut[["short_code", "std_code", "name", "expiry_ym", "multiplier", "und_code", "und_name"]].reset_index(
        drop=True
    )
