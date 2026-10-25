"""03-4. 키움 REST 현물 일봉과 KIS 현물 일봉 교차검증 (선택 심화, 조회 전용).

선행: 03_collect_daily.py (KIS 현물 일봉)
실행: uv run python examples/chapter03/05_kiwoom_crosscheck.py [005930] [시작일=20250101]
"""

import sys

from dotenv import load_dotenv

from hedging.data.collect import load_spot
from hedging.data.kiwoom import KiwoomClient

load_dotenv()
und = sys.argv[1] if len(sys.argv) > 1 else "005930"
start = sys.argv[2] if len(sys.argv) > 2 else "20250101"

kis = load_spot(und)
end = kis["date"].max().strftime("%Y%m%d")
kiw = KiwoomClient.from_env("real").stock_daily(und, start, end)
m = kis.merge(kiw, on="date", suffixes=("_kis", "_kiw"), how="outer", indicator=True)
m = m[m["date"] >= kiw["date"].min()]
both = m[m["_merge"] == "both"]

print(f"{und} {start} ~ {end}")
print(f"  KIS {int((m['_merge'] != 'right_only').sum())}일, 키움 {int((m['_merge'] != 'left_only').sum())}일, 공통 {len(both)}일")
print(f"  한쪽에만 있는 날짜: {m.loc[m['_merge'] != 'both', 'date'].dt.strftime('%Y-%m-%d').tolist()[:10]}")
for col in ("open", "high", "low", "close", "volume"):
    diff = (both[f"{col}_kis"] - both[f"{col}_kiw"]).abs()
    print(f"  {col:<6} 불일치 {int((diff > 0).sum()):>4}일  최대 차이 {diff.max():,.0f}")
if (both["volume_kis"] != both["volume_kiw"]).any():
    r = (both["volume_kiw"] / both["volume_kis"]).describe()
    print(f"  거래량 비율(키움/KIS) 중앙값 {r['50%']:.4f}, 최소 {r['min']:.4f}, 최대 {r['max']:.4f}")
    sample = both[both["volume_kis"] != both["volume_kiw"]].tail(3)
    print(sample[["date", "close_kis", "close_kiw", "volume_kis", "volume_kiw"]].to_string(index=False))
