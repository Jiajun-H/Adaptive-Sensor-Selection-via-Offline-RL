"""
Compose existing IQL sensor-selection heatmaps into a paper-ready SVG.

The source heatmaps were generated during prior evaluations and already encode
the policy-selected sensor masks. This script does not infer or fabricate mask
values; it only creates a self-contained figure panel from those PNG files.
"""

from __future__ import annotations

import argparse
import base64
import struct
from pathlib import Path
from xml.sax.saxutils import escape


DEFAULT_SOURCES = [
    Path("figures/repeated/kfold_seed42_fold0/sensor_heatmap_LSTM+IQL.png"),
    Path("figures/repeated/kfold_seed42_fold1/sensor_heatmap_LSTM+IQL.png"),
    Path("figures/repeated/kfold_seed42_fold2/sensor_heatmap_LSTM+IQL.png"),
]

FONT = "'Aptos','Segoe UI','Liberation Sans','DejaVu Sans',Arial,sans-serif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose IQL sensor-selection heatmaps.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/sensor_selection_paper"))
    parser.add_argument("--title", default="IQL Sensor Selection Patterns Across Held-out Folds")
    parser.add_argument("--source", type=Path, action="append", help="PNG heatmap path. Repeat for multiple panels.")
    return parser.parse_args()


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def panel_label(index: int, path: Path) -> str:
    for part in path.parts:
        if part.startswith("kfold_seed"):
            return part.replace("kfold_seed42_", "").replace("_", " ")
    return f"fold {index}"


def write_svg(output: Path, sources: list[Path], title: str) -> None:
    missing = [p for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source heatmap(s): " + ", ".join(str(p) for p in missing))

    page_w = 2400
    margin_x = 58
    top = 122
    panel_gap = 34
    panel_pad = 16
    label_h = 32
    bottom_note_h = 58
    panel_w = (page_w - 2 * margin_x - panel_gap * (len(sources) - 1)) / len(sources)
    image_w = panel_w - 2 * panel_pad

    panels: list[tuple[Path, int, int, float, float, float]] = []
    max_panel_h = 0.0
    for i, path in enumerate(sources):
        w, h = png_size(path)
        scale = image_w / w
        image_h = h * scale
        x = margin_x + i * (panel_w + panel_gap)
        panel_h = label_h + image_h + 2 * panel_pad
        panels.append((path, w, h, x, image_h, panel_h))
        max_panel_h = max(max_panel_h, panel_h)
    page_h = int(top + max_panel_h + bottom_note_h + 36)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w}" height="{page_h}" viewBox="0 0 {page_w} {page_h}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="{page_w / 2:.1f}" y="48" font-family="{FONT}" font-size="30" font-weight="800" text-anchor="middle" fill="#1F2430">{escape(title)}</text>',
        f'<text x="{page_w / 2:.1f}" y="78" font-family="{FONT}" font-size="15" font-weight="500" text-anchor="middle" fill="#5E6878">Red cells indicate selected sensors; pale cells indicate inactive sensors; blue dashed line marks fire onset.</text>',
    ]

    for i, (path, w, h, x, image_h, panel_h) in enumerate(panels):
        label = panel_label(i, path)
        parts.append(
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{panel_w:.1f}" height="{max_panel_h:.1f}" '
            'rx="8" fill="#FAFBFC" stroke="#E3E8F0" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{x + panel_pad:.1f}" y="{top + 22:.1f}" font-family="{FONT}" font-size="18" font-weight="800" '
            f'text-anchor="start" fill="#1F2430">({chr(97 + i)}) {escape(label.title())}</text>'
        )
        parts.append(
            f'<image x="{x + panel_pad:.1f}" y="{top + label_h + panel_pad:.1f}" width="{image_w:.1f}" height="{image_h:.1f}" '
            f'href="{data_uri(path)}" preserveAspectRatio="none"/>'
        )

    note_y = top + max_panel_h + 34
    parts.append(
        f'<text x="{page_w / 2:.1f}" y="{note_y:.1f}" font-family="{FONT}" font-size="16" font-weight="600" '
        'text-anchor="middle" fill="#4E596A">'
        "Sensors are frequency-ranked within each fold; persistent red bands mark core channels and intermittent stripes mark state-dependent substitutions."
        "</text>"
    )

    parts.append("</svg>\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def write_notes(output_dir: Path, figure_path: Path, sources: list[Path]) -> None:
    notes = [
        "# Sensor Selection Figure Notes",
        "",
        f"- Main figure: `{figure_path.as_posix()}`",
        "- Source heatmaps:",
    ]
    notes.extend(f"  - `{p.as_posix()}`" for p in sources)
    notes.extend(
        [
            "",
            "## Recommended Caption",
            "",
            "IQL sensor-selection heatmaps on the held-out folds. Rows denote the most frequently selected sensors within each fold and columns denote evaluation time steps. Red cells indicate selected sensors, pale cells indicate inactive sensors, and the blue dashed line marks the fire onset. The block-like horizontal bands show that the policy maintains a stable core sensor subset, while the intermittent vertical stripes indicate limited state-dependent substitutions rather than random reconfiguration.",
            "",
            "## Interpretation",
            "",
            "- This figure adds information not contained in the metric tables: it shows which selection behavior produced the reported F1/lead-time/switching results.",
            "- The top rows are frequently active across long intervals, suggesting a core sensor set learned by IQL.",
            "- Lower rows show intermittent activation, suggesting secondary sensors used for local or stage-specific evidence.",
            "- The repeated block structure across folds supports the claim that IQL is not simply switching arbitrarily; it uses stable cores plus sparse adaptive changes.",
            "",
            "## Data Caveat",
            "",
            "This composed figure uses the existing evaluation heatmaps. For quantitative sensor frequency and modality statistics, run `export_sensor_selection_masks.py` in the project environment where torch/numpy/pandas/matplotlib are available.",
            "",
        ]
    )
    (output_dir / "sensor_selection_figure_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sources = args.source if args.source else DEFAULT_SOURCES
    output = args.output_dir / "01_iql_sensor_selection_heatmaps_by_fold.svg"
    write_svg(output, sources, args.title)
    write_notes(args.output_dir, output, sources)
    print(f"Saved {output}")
    print(f"Wrote {args.output_dir / 'sensor_selection_figure_notes.md'}")


if __name__ == "__main__":
    main()

