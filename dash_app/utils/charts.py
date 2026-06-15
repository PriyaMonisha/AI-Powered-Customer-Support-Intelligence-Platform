# filename: dash_app/utils/charts.py
# purpose:  Palette helpers (locked visualization rules) + static chart helpers for
#           the /charts/<file> route.
# version:  1.0

import matplotlib
import plotly.express as px

from config import CHARTS_DIR

ALLOWED_CHART_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _mpl_cmap_to_hex(cmap_name: str, n: int) -> list[str]:
    cmap = matplotlib.colormaps[cmap_name]
    return [matplotlib.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def multiclass_colors(n: int) -> list[str]:
    """Discrete Plotly Set2 palette, cycled for n > len(palette)."""
    palette = px.colors.qualitative.Set2
    return [palette[i % len(palette)] for i in range(n)]


def binary_colors() -> list[str]:
    """2 colors from Dark2_r for binary comparisons."""
    return _mpl_cmap_to_hex("Dark2_r", 2)


def histogram_color() -> str:
    """1 color from Accent_r for histograms/distributions."""
    return _mpl_cmap_to_hex("Accent_r", 1)[0]


def chart_exists(filename: str) -> bool:
    return (CHARTS_DIR / filename).exists()


def chart_url(filename: str) -> str:
    return f"/charts/{filename}"
