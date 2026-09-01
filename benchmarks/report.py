"""Create a dependency-free SVG trend plot from stateful benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_svg(data: dict) -> str:
    """Render cycle times as a small standalone SVG line chart."""

    cycles = data["cycles"]
    values = [float(item["seconds"]) for item in cycles]
    width, height, pad = 720, 320, 48
    maximum = max(values) if values else 1.0
    maximum = maximum or 1.0
    x_span = max(1, len(values) - 1)
    points = []
    for index, value in enumerate(values):
        x = pad + (width - 2 * pad) * index / x_span
        y = height - pad - (height - 2 * pad) * value / maximum
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    ratio = float(data.get("degradation_ratio", 0.0))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="black"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="black"/>
<polyline fill="none" stroke="black" stroke-width="2" points="{polyline}"/>
<text x="{pad}" y="24" font-family="sans-serif" font-size="16">Stateful workload cycle time; degradation ratio {ratio:.3f}</text>
<text x="{width / 2 - 30}" y="{height - 10}" font-family="sans-serif" font-size="12">cycle</text>
<text x="8" y="{pad}" font-family="sans-serif" font-size="12">{maximum:.3f}s</text>
</svg>\n'''


def main() -> None:
    """Read benchmark JSON and write a standalone SVG report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    Path(args.output).write_text(render_svg(data))


if __name__ == "__main__":
    main()
