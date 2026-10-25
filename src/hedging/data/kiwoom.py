"""키움증권 REST API 조회 전용 클라이언트 (3장 교차검증, 선택).

- 토큰: POST /oauth2/token (appkey, secretkey)
- 주식 일봉: POST /api/dostk/chart, api-id ka10081. 한 번에 최대 600행,
  응답 헤더 cont-yn=Y 이면 next-key 를 요청 헤더에 넣어 이어서 받는다.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

HOSTS = {"real": "https://api.kiwoom.com", "paper": "https://mockapi.kiwoom.com"}


class KiwoomClient:
    def __init__(self, app_key: str, app_secret: str, env: str = "real"):
        self.host = HOSTS[env]
        self.app_key, self.app_secret = app_key, app_secret
        self._token: str | None = None

    @classmethod
    def from_env(cls, env: str = "real") -> "KiwoomClient":
        p = env.upper()
        key, secret = os.getenv(f"KIWOOM_{p}_APP_KEY"), os.getenv(f"KIWOOM_{p}_APP_SECRET")
        if not key or not secret:
            raise RuntimeError(f"KIWOOM_{p}_APP_KEY / KIWOOM_{p}_APP_SECRET 가 설정되지 않았습니다 (.env 확인)")
        return cls(key, secret, env)

    def token(self) -> str:
        if not self._token:
            body = requests.post(f"{self.host}/oauth2/token", timeout=10, json={
                "grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}).json()
            if "token" not in body:
                raise RuntimeError(f"키움 토큰 발급 실패: {body.get('return_code')} {body.get('return_msg')}")
            self._token = body["token"]
        return self._token

    def post(self, path: str, api_id: str, body: dict, next_key: str | None = None) -> tuple[dict, dict]:
        headers = {"Content-Type": "application/json;charset=UTF-8", "authorization": f"Bearer {self.token()}",
                   "api-id": api_id}
        if next_key:
            headers.update({"cont-yn": "Y", "next-key": next_key})
        r = requests.post(f"{self.host}{path}", headers=headers, json=body, timeout=10)
        data = r.json()
        if data.get("return_code") != 0:
            raise RuntimeError(f"[{api_id}] {data.get('return_code')} {data.get('return_msg')}")
        return data, dict(r.headers)

    def stock_daily(self, code: str, start: str, end: str, adjusted: bool = False) -> pd.DataFrame:
        """주식 일봉(ka10081). end 기준으로 과거로 거슬러 받다가 start 이전에 닿으면 멈춘다."""
        rows, key = [], None
        while True:
            data, h = self.post("/api/dostk/chart", "ka10081",
                                {"stk_cd": code, "base_dt": end, "upd_stkpc_tp": "1" if adjusted else "0"}, key)
            page = data.get("stk_dt_pole_chart_qry") or []
            rows += page
            if not page or min(r["dt"] for r in page) <= start or h.get("cont-yn") != "Y":
                break
            key = h.get("next-key")
            time.sleep(0.3)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.rename(columns={"dt": "date", "open_pric": "open", "high_pric": "high", "low_pric": "low",
                                "cur_prc": "close", "trde_qty": "volume"})
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce").abs()  # 가격에 등락 부호가 붙어 올 수 있다
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        return df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date") \
            .sort_values("date").reset_index(drop=True)
