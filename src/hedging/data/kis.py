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
from datetime import datetime
from pathlib import Path

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
