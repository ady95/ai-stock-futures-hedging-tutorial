"""한국투자증권(KIS) Open API 조회 전용 클라이언트 (2~3장).

이 모듈은 시세 조회만 한다. 주문 기능은 10장에서 별도의 안전장치와 함께 다룬다.

실측으로 확인한 제약 (2026-10)
- 접근 토큰은 1분에 1회만 새로 발급된다(EGW00133). 토큰은 24시간 유효하므로 파일에 캐시해 재사용한다.
- 모의투자 서버는 초당 호출 한도가 낮다(EGW00201). 호출 간격을 둔다.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

HOSTS = {
    "real": "https://openapi.koreainvestment.com:9443",
    "paper": "https://openapivts.koreainvestment.com:29443",
}
MIN_INTERVAL = {"real": 0.1, "paper": 1.0}  # 초


class KisClient:
    def __init__(self, app_key: str, app_secret: str, env: str = "real", cache_dir: str | Path = ".cache"):
        if env not in HOSTS:
            raise ValueError(f"env must be one of {list(HOSTS)}")
        self.app_key, self.app_secret, self.env = app_key, app_secret, env
        self.host = HOSTS[env]
        self.cache_file = Path(cache_dir) / f"kis_token_{env}.json"
        self._last_call = 0.0
        self._token: str | None = None

    @classmethod
    def from_env(cls, env: str = "real", **kw) -> "KisClient":
        """환경변수(KIS_REAL_APP_KEY 등)에서 키를 읽는다. .env는 호출 쪽에서 load_dotenv()로 불러온다."""
        p = env.upper()
        key, secret = os.getenv(f"KIS_{p}_APP_KEY"), os.getenv(f"KIS_{p}_APP_SECRET")
        if not key or not secret:
            raise RuntimeError(f"KIS_{p}_APP_KEY / KIS_{p}_APP_SECRET 가 설정되지 않았습니다 (.env 확인)")
        return cls(key, secret, env, **kw)

    # ---- 인증 -------------------------------------------------------------
    def token(self) -> str:
        if self._token:
            return self._token
        if self.cache_file.exists():
            cached = json.loads(self.cache_file.read_text(encoding="utf-8"))
            expires = datetime.strptime(cached["expires"], "%Y-%m-%d %H:%M:%S")
            if (expires - datetime.now()).total_seconds() > 600:  # 10분 이상 남았으면 재사용
                self._token = cached["token"]
                return self._token
        r = requests.post(
            f"{self.host}/oauth2/tokenP",
            json={"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret},
            timeout=10,
        )
        body = r.json()
        if "access_token" not in body:
            raise RuntimeError(f"토큰 발급 실패: {body.get('error_code')} {body.get('error_description')}")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(
            json.dumps({"token": body["access_token"], "expires": body["access_token_token_expired"]}),
            encoding="utf-8",
        )
        self._token = body["access_token"]
        return self._token

    # ---- 공통 GET -----------------------------------------------------------
    def get(self, path: str, tr_id: str, params: dict) -> dict:
        wait = MIN_INTERVAL[self.env] - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        r = requests.get(f"{self.host}{path}", headers=headers, params=params, timeout=10)
        self._last_call = time.time()
        body = r.json()
        if body.get("rt_cd") != "0":
            raise RuntimeError(f"[{tr_id}] {body.get('msg_cd')} {body.get('msg1')}")
        return body

    # ---- 시세 ---------------------------------------------------------------
    def stock_price(self, code: str) -> dict:
        """국내주식 현재가 (tr_id FHKST01010100)."""
        return self.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )["output"]

    def stock_futures_price(self, code: str) -> dict:
        """주식선물 현재가 (tr_id FHMIF10000000, 시장구분 JF=주식선물). output1을 돌려준다."""
        return self.get(
            "/uapi/domestic-futureoption/v1/quotations/inquire-price",
            "FHMIF10000000",
            {"FID_COND_MRKT_DIV_CODE": "JF", "FID_INPUT_ISCD": code},
        )["output1"]

    # ---- 일봉 (3장) -----------------------------------------------------------
    def _paged_daily(self, fetch, start: str, end: str) -> list[dict]:
        """한 번에 최대 100행만 오므로 종료일을 가장 이른 날짜 전날로 당기며 반복한다."""
        rows: list[dict] = []
        d2 = end
        while d2 >= start:
            page = [r for r in fetch(start, d2) if r.get("stck_bsop_date")]
            rows += page
            if len(page) < 100:
                break
            earliest = datetime.strptime(min(r["stck_bsop_date"] for r in page), "%Y%m%d").date()
            d2 = (earliest - timedelta(days=1)).strftime("%Y%m%d")
        return rows

    def stock_futures_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        """주식선물 일봉 (tr_id FHKIF03020100). 날짜는 YYYYMMDD."""

        def fetch(d1, d2):
            return self.get(
                "/uapi/domestic-futureoption/v1/quotations/inquire-daily-fuopchartprice",
                "FHKIF03020100",
                {"FID_COND_MRKT_DIV_CODE": "JF", "FID_INPUT_ISCD": code, "FID_INPUT_DATE_1": d1,
                 "FID_INPUT_DATE_2": d2, "FID_PERIOD_DIV_CODE": "D"},
            ).get("output2") or []

        rows = self._paged_daily(fetch, start, end)
        cols = {"stck_bsop_date": "date", "futs_oprc": "open", "futs_hgpr": "high", "futs_lwpr": "low",
                "futs_prpr": "close", "acml_vol": "volume", "acml_tr_pbmn": "value", "mod_yn": "mod_yn"}
        return _to_frame(rows, cols)

    def stock_daily(self, code: str, start: str, end: str, adjusted: bool = False) -> pd.DataFrame:
        """국내주식 일봉 (tr_id FHKST03010100).

        베이시스 계산에는 선물과 같은 기준인 원주가(adjusted=False)를 쓴다. 수정주가는 과거 가격이
        액면분할 등에 맞춰 바뀌므로 그날의 선물가격과 비교할 수 없다.
        """

        def fetch(d1, d2):
            return self.get(
                "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                "FHKST03010100",
                {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code, "FID_INPUT_DATE_1": d1,
                 "FID_INPUT_DATE_2": d2, "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0" if adjusted else "1"},
            ).get("output2") or []

        rows = self._paged_daily(fetch, start, end)
        cols = {"stck_bsop_date": "date", "stck_oprc": "open", "stck_hgpr": "high", "stck_lwpr": "low",
                "stck_clpr": "close", "acml_vol": "volume", "acml_tr_pbmn": "value", "mod_yn": "mod_yn"}
        return _to_frame(rows, cols)


    # ---- 배당 (4장) ---------------------------------------------------------
    def dividend_schedule(self, code: str, start: str, end: str) -> pd.DataFrame:
        """예탁원정보 배당일정 (tr_id HHKDB669102C0).

        record_date(배당기준일), dps(주당 현금배당금), kind(분기/중간/결산), pay_date.
        아직 결정되지 않은 배당은 dps=0 으로 온다.
        """
        b = self.get("/uapi/domestic-stock/v1/ksdinfo/dividend", "HHKDB669102C0",
                     {"CTS": "", "GB1": "0", "F_DT": start, "T_DT": end, "SHT_CD": code, "HIGH_GB": ""})
        rows = b.get("output1") or []
        df = pd.DataFrame([{
            "record_date": pd.to_datetime(r["record_date"], format="%Y%m%d"),
            "dps": float(r["per_sto_divi_amt"] or 0),
            "kind": r["divi_kind"],
            "pay_date": pd.to_datetime(r["divi_pay_dt"], format="%Y/%m/%d") if r.get("divi_pay_dt") else pd.NaT,
        } for r in rows if r.get("stk_kind", "보통") == "보통"])
        return df.sort_values("record_date").reset_index(drop=True) if len(df) else df


def _to_frame(rows: list[dict], cols: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(cols)).rename(columns=cols)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in ("open", "high", "low", "close", "volume", "value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")
