"""KRX Open API — 주식선물 일별매매정보 (3장).

- 인증키(AUTH_KEY 헤더)만으로는 호출되지 않는다. 사이트에서 서비스별 'API 이용신청' 승인이 필요하다(실측).
- 하루치 '전 종목' 단면을 돌려준다. 유가: drv/eqsfu_stk_bydd_trd(2010-01-04~), 코스닥: drv/eqkfu_ksq_bydd_trd(2015-08-03~)
- 응답에는 월물 간 스프레드 행이 섞여 있어 선물만 골라낸다.
- ISU_CD(8자리)는 월을 한 글자(1~9, A, B, C)로 쓴다. KIS 단축코드(6자리)는 월을 두 자리 숫자로 쓴다.
  예) A116B000 ↔ A11611, 111Q1000 ↔ 111Q01

원시 응답은 data/raw/krx/ 에만 두고 커밋하지 않는다(재배포 약관).
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data-dbg.krx.co.kr/svc/apis/"
SERVICES = {"kospi": "drv/eqsfu_stk_bydd_trd", "kosdaq": "drv/eqkfu_ksq_bydd_trd"}
RAW = Path("data/raw/krx")
MONTH_CHAR = {**{str(i): i for i in range(1, 10)}, "A": 10, "B": 11, "C": 12}
NUM = ["TDD_CLSPRC", "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "SPOT_PRC", "SETL_PRC",
       "ACC_TRDVOL", "ACC_TRDVAL", "ACC_OPNINT_QTY"]


def kis_short_code(isu_cd: str) -> str:
    """KRX 8자리 종목코드 → KIS 6자리 단축코드."""
    return f"{isu_cd[:4]}{MONTH_CHAR[isu_cd[4]]:02d}"


class KrxClient:
    def __init__(self, auth_key: str):
        self.auth_key = auth_key

    @classmethod
    def from_env(cls) -> "KrxClient":
        key = os.getenv("KRX_API_KEY")
        if not key:
            raise RuntimeError("KRX_API_KEY 가 설정되지 않았습니다 (.env 확인)")
        return cls(key)

    def raw(self, market: str, bas_dd: str, retries: int = 3) -> list[dict]:
        for i in range(retries):
            r = requests.get(BASE + SERVICES[market], params={"basDd": bas_dd},
                             headers={"AUTH_KEY": self.auth_key}, timeout=60)
            if r.status_code == 401:
                raise RuntimeError("401 Unauthorized: 인증키 또는 서비스별 이용신청 승인을 확인하세요")
            if r.ok:
                return r.json().get("OutBlock_1") or []
            time.sleep(2 * (i + 1))
        r.raise_for_status()
        return []

    def stock_futures_daily(self, market: str, bas_dd: str) -> pd.DataFrame:
        """하루치 주식선물 단면 (스프레드 제외)."""
        rows = [x for x in self.raw(market, bas_dd) if re.search(r"F \d{6}", x["ISU_NM"])]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for c in NUM:
            df[c] = pd.to_numeric(df[c].replace("", None), errors="coerce")
        out = pd.DataFrame({
            "date": pd.to_datetime(df["BAS_DD"], format="%Y%m%d"),
            "market": market,
            "und_name": df["PROD_NM"].str.replace(r"\s*선물$", "", regex=True).str.strip(),
            "isu_cd": df["ISU_CD"],
            "short_code": df["ISU_CD"].map(kis_short_code),
            "expiry_ym": df["ISU_NM"].str.extract(r"F (\d{6})")[0],
            "open": df["TDD_OPNPRC"], "high": df["TDD_HGPRC"], "low": df["TDD_LWPRC"],
            "close": df["TDD_CLSPRC"], "settle": df["SETL_PRC"], "spot": df["SPOT_PRC"],
            "volume": df["ACC_TRDVOL"], "value": df["ACC_TRDVAL"], "open_interest": df["ACC_OPNINT_QTY"],
        })
        return out


def collect_days(client: KrxClient, market: str, days: list[str], workers: int = 4, root: Path = RAW) -> int:
    """거래일 목록을 날짜별 Parquet으로 저장한다. 이미 받은 날은 건너뛴다. 새로 받은 일수를 돌려준다."""
    out_dir = root / market
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (out_dir / f"{d}.parquet").exists()]

    def one(d: str) -> int:
        df = client.stock_futures_daily(market, d)
        df.to_parquet(out_dir / f"{d}.parquet", index=False)
        return 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(one, todo))


def load_krx(market: str, root: Path = RAW) -> pd.DataFrame:
    files = sorted((root / market).glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def universe(df: pd.DataFrame) -> pd.DataFrame:
    """기초자산별 첫·마지막 등장일과 현재 상장 여부. ETF 선물은 따로 표시한다."""
    last_day = df["date"].max()
    u = df.groupby(["market", "und_name"])["date"].agg(first="min", last="max").reset_index()
    u["is_etf"] = u["und_name"].str.startswith("ETF")
    u["listed_now"] = u["last"] == last_day
    return u.sort_values(["market", "first", "und_name"]).reset_index(drop=True)


def listed_on(df: pd.DataFrame, day: str | pd.Timestamp, stocks_only: bool = True) -> list[str]:
    """그날 실제로 상장되어 있던 기초자산 목록 (백테스트의 생존자 편향 방지용)."""
    s = df[df["date"] == pd.Timestamp(day)]
    if stocks_only:
        s = s[~s["und_name"].str.startswith("ETF")]
    return sorted(s["und_name"].unique())


def yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """연도별로 한 번이라도 거래 가능했던 개별주식 기초자산 수."""
    s = df[~df["und_name"].str.startswith("ETF")]
    return s.groupby([s["date"].dt.year.rename("year"), "market"])["und_name"].nunique().unstack(fill_value=0)
