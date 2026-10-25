"""03-1. KIS 실전/모의 인증과 토큰 캐시 확인 (조회 전용).

토큰은 1분에 1번만 새로 발급되고 24시간 유효하다. 같은 스크립트를 두 번 실행해 보면
두 번째에는 캐시 파일(.cache/kis_token_<env>.json)을 재사용한다.
실행: uv run python examples/chapter03/01_auth_token_cache.py
"""

import json
import time

from dotenv import load_dotenv

from hedging.data.kis import HOSTS, KisClient

load_dotenv()
for env in ("real", "paper"):
    try:
        c = KisClient.from_env(env)
    except RuntimeError as e:
        print(f"[{env}] 건너뜀: {e}")
        continue
    cached = c.cache_file.exists()
    t0 = time.time()
    c.token()
    expires = json.loads(c.cache_file.read_text(encoding="utf-8"))["expires"]
    print(f"[{env}] 호스트 {HOSTS[env]}")
    print(f"  토큰: {'캐시 재사용' if cached else '새로 발급'} ({time.time() - t0:.2f}초), 만료 {expires}")

    # 연속 호출 3번 — 모의 서버는 호출 간격(1초)을 두도록 클라이언트가 조절한다
    t0 = time.time()
    for _ in range(3):
        price = c.stock_price("005930")["stck_prpr"]
    print(f"  현재가 3회 조회 {time.time() - t0:.1f}초, 삼성전자 {int(price):,}원")
