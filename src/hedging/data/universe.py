"""여러 기초자산으로 넓히기: KRX 주식선물 + KIS 현물·배당 (6장).

- 선물: KRX 주식선물 일별매매정보(3-2). 상장 기간이 데이터에 그대로 들어 있어 생존자 편향이 없다
- 현물: KRX 데이터에는 기초주식 코드가 없으므로, KIS 주식 종목 목록 파일의 종목명으로 코드를 찾는다
  (이름이 바뀌었거나 상장폐지된 종목은 찾지 못할 수 있다 → 매핑 실패 목록을 남긴다)
- 현물 일봉·배당은 KIS API로 받는다(종목당 약 20회 호출)
"""

from __future__ import annotations

import io
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from hedging.data.collect import collect_dividends, collect_spot
from hedging.data.dataset import front_month_series
from hedging.data.kis import KisClient
from hedging.features.basis import add_features
from hedging.features.fair_value import fair_value
from hedging.strategies.basis_rules import attach_opens

MASTER_URLS = {
    "kospi": ("https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip", 228),
    "kosdaq": ("https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip", 222),
}
RAW = Path("data/raw")
OUT = Path("data/processed/universe")


def stock_master(market: str, raw: Path = RAW) -> pd.DataFrame:
    """KIS 주식 종목 목록(단축코드·표준코드·한글명). 공식 저장소 stocks_info 의 고정폭 규칙을 따른다."""
    url, tail = MASTER_URLS[market]
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        z.extract(name, raw)
    rows = []
    for line in (raw / name).read_text(encoding="cp949").splitlines():
        head = line[: len(line) - tail]
        rows.append({"code": head[0:9].strip(), "std_code": head[9:21].strip(), "name": head[21:].strip()})
    df = pd.DataFrame(rows)
    df["market"] = market
    return df


def _norm(name: str) -> str:
    return re.sub(r"\s+", "", name).upper()


# 선물 상품명과 주식 종목명이 다르거나, 주식이 상장폐지되어 종목 목록에 없는 경우의 별칭.
# 정확성은 build_features()의 '현물 종가 = KRX 기초자산 가격' 일치율로 검증한다.
ALIASES = {
    "JYP Ent": "035900", "와이지엔터": "122870", "원익큐엔씨": "074600", "PSK홀딩스": "031980",
    "SK머티리얼즈": "036490", "셀트리온헬스케어": "091990", "HDC 현대산업개발": "294870",
    "HD현대미포": "010620", "HD현대인프라코어": "042670", "메리츠증권": "008560",
    "오렌지라이프": "079440", "쌍용C&E": "003410",
}


def map_underlyings(und_names: list[str], masters: pd.DataFrame,
                    fut_master: pd.DataFrame | None = None, krx: pd.DataFrame | None = None) -> pd.DataFrame:
    """KRX 기초자산명 → 현물 종목코드.

    1) 선물 종목코드가 KIS 주식선물 종목 목록(유가, 현재 상장)에 있으면 그 기초주식 코드
    2) 주식 종목명 일치(공백·대소문자 무시)  3) ALIASES
    """
    m = masters[masters["code"].str.fullmatch(r"\d{6}")].copy()  # 보통주 6자리 코드만
    m["key"] = m["name"].map(_norm)
    by_name = m.drop_duplicates("key").set_index("key")["code"]
    by_fut = {}
    if fut_master is not None and krx is not None:
        code_to_und = dict(zip(fut_master["short_code"], fut_master["und_code"]))
        for n, codes in krx.groupby("und_name")["short_code"].unique().items():
            hit = [code_to_und[c] for c in codes if c in code_to_und]
            if hit:
                by_fut[n] = hit[0]
    rows = []
    for n in und_names:
        k = _norm(n)
        if n in by_fut:
            rows.append((n, by_fut[n], "선물코드"))
        elif k in by_name:
            rows.append((n, by_name[k], "종목명"))
        elif n in ALIASES:
            rows.append((n, ALIASES[n], "별칭"))
        else:
            rows.append((n, None, "실패"))
    return pd.DataFrame(rows, columns=["und_name", "code", "how"])


def collect_universe(make_client, mapping: pd.DataFrame, start: str, end: str,
                     raw: Path = RAW / "kis", workers: int = 6) -> pd.DataFrame:
    """매핑된 종목의 현물 일봉(원주가)과 배당을 받는다. 이미 받은 종목은 건너뛴다.

    make_client: KisClient 를 만드는 함수. 스레드마다 클라이언트를 따로 만들어 동시에 받는다
    (토큰은 파일 캐시를 함께 쓰므로 새로 발급하지 않는다). 종목당 호출이 약 18회라 순차로는 오래 걸린다.
    """
    local = threading.local()

    def one(code: str) -> dict:
        if not hasattr(local, "client"):
            local.client = make_client()
        client: KisClient = local.client
        spot_file = raw / "stock_daily" / f"{code}.parquet"
        div_file = raw / "dividends" / f"{code}.parquet"
        try:
            if spot_file.exists():
                n_spot = len(pd.read_parquet(spot_file))
            else:
                n_spot = len(collect_spot(client, code, start, end, root=raw))
            if not div_file.exists():
                collect_dividends(client, code, start, end, root=raw)
            return {"code": code, "spot_rows": n_spot, "ok": True}
        except Exception as e:  # 한 종목 실패가 전체를 멈추지 않게
            return {"code": code, "spot_rows": 0, "ok": False, "error": str(e)[:80]}

    codes = list(mapping["code"].dropna().unique())
    make_client().token()  # 스레드를 띄우기 전에 토큰을 한 번 받아 캐시해 둔다(동시 발급 시 EGW00133)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return pd.DataFrame(list(ex.map(one, codes)))


def krx_futures_as_kis(krx: pd.DataFrame, und_names: str | list[str]) -> pd.DataFrame:
    """KRX 행을 front_month_series()가 기대하는 형식(code, expiry_ym, date, open, close, volume, settle)으로.

    선물 상품명이 바뀐 종목은 여러 이름을 함께 넘겨 하나의 시계열로 합친다.
    """
    names = [und_names] if isinstance(und_names, str) else list(und_names)
    k = krx[krx["und_name"].isin(names)]
    return k.rename(columns={"short_code": "code"})[
        ["code", "expiry_ym", "date", "open", "close", "volume", "settle", "open_interest"]].reset_index(drop=True)


def build_features(krx: pd.DataFrame, und_name: str | list[str], code: str, raw: Path = RAW / "kis",
                   out: Path = OUT) -> pd.DataFrame | None:
    """한 기초주식의 최근월물 시계열 → 이론가격·괴리 피처. 정산가·시가 열도 붙여 저장한다."""
    names = [und_name] if isinstance(und_name, str) else list(und_name)
    fut = krx_futures_as_kis(krx, names)
    spot_file = raw / "stock_daily" / f"{code}.parquet"
    if fut.empty or not spot_file.exists():
        return None
    spot = pd.read_parquet(spot_file)
    spot = spot[spot["date"] <= fut["date"].max()]
    divs_file = raw / "dividends" / f"{code}.parquet"
    divs = pd.read_parquet(divs_file) if divs_file.exists() else None
    series = front_month_series(fut, spot, roll_days=0)
    series = series[series["date"].isin(fut["date"])]  # 선물이 상장되어 있던 날만
    if series.empty:
        return None
    fv = add_features(fair_value(series, divs if divs is not None and len(divs) else None))
    st = fut.set_index(["code", "date"])["settle"]
    fv["front_settle"] = [st.get((c, d)) for c, d in zip(fv["front_code"], fv["date"])]
    # 매핑 검증: KRX가 주는 기초자산 가격(SPOT_PRC)은 현물 종가와 같다(3-2)
    krx_spot = krx[krx["und_name"].isin(names)].groupby("date")["spot"].first()
    fv["krx_spot"] = fv["date"].map(krx_spot)
    fv = attach_opens(fv, fut, spot)  # 5장 전략·6장 장부가 쓰는 시가 열
    fv.insert(0, "und_name", "/".join(names))
    fv.insert(1, "und_code", code)
    out.mkdir(parents=True, exist_ok=True)
    fv.to_parquet(out / f"{code}_features.parquet", index=False)
    return fv


def load_universe(valid_only: bool = True, out: Path = OUT) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """매핑 표와 기초자산별 피처(코드 → DataFrame). valid_only 면 현물 종가 검증을 통과한 종목만."""
    mapping = pd.read_parquet("data/processed/universe_mapping.parquet")
    if valid_only:
        mapping = mapping[mapping["valid"].fillna(False)]
    feats = {}
    for code in mapping["code"].dropna().unique():
        f = out / f"{code}_features.parquet"
        if f.exists():
            feats[code] = pd.read_parquet(f)
    return mapping, feats
