"""matplotlib 한글 폰트 설정. OS마다 설치된 폰트가 달라 후보 중 설치된 첫 폰트를 고른다."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]


def setup() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in KOREAN_FONTS if f in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    return font
