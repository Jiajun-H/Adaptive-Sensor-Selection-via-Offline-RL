"""
Batch-export workbook data from Origin `.opju` projects to Excel files.

Requirements:
1. Origin/OriginPro installed on Windows with COM automation available.
2. `pywin32` installed in the conda environment used to run this script:
      conda install -n rl -c conda-forge pywin32

Usage:
    conda run -n rl python convert_opju_to_excel.py
    conda run -n rl python convert_opju_to_excel.py --input-root "试验数据记录-评审" --output-root "converted_excel" --overwrite
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path


LOGGER = logging.getLogger("convert_opju_to_excel")
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively convert workbook pages in .opju projects to Excel files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("试验数据记录-评审"),
        help="Root directory to search for .opju files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("converted_excel"),
        help="Root directory for exported Excel files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.opju",
        help="Glob pattern searched under --input-root. Default: *.opju",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported Excel files.",
    )
    parser.add_argument(
        "--cell-formats",
        action="store_true",
        help="Keep Origin cell formatting when exporting if possible.",
    )
    parser.add_argument(
        "--show-origin",
        action="store_true",
        help="Show the Origin window during conversion.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Small wait after loading/exporting each project. Default: 0.5",
    )
    args = parser.parse_args()

    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()

    if not args.input_root.exists():
        parser.error(f"Input root does not exist: {args.input_root}")

    return args


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def sanitize_filename(name: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", name).strip()
    cleaned = cleaned.rstrip(". ")
    return cleaned or "unnamed"


def lt_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def import_win32com():
    try:
        import win32com.client as win32  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "未找到 win32com。请使用 conda 安装：`conda install -n rl -c conda-forge pywin32`。"
        ) from exc
    return win32


def dispatch_origin(win32, show_origin: bool):
    try:
        app = win32.gencache.EnsureDispatch("Origin.ApplicationSI")
    except Exception as exc:
        raise RuntimeError(
            "无法连接 Origin COM。请确认 Origin 已安装并已正确注册自动化接口。"
        ) from exc

    try:
        app.Visible = bool(show_origin)
    except Exception:
        pass

    return app


def workbook_export_prefix(page, index: int) -> str:
    long_name = str(getattr(page, "LongName", "") or "").strip()
    short_name = str(getattr(page, "Name", "") or "").strip()
    base = long_name or short_name or f"book_{index:02d}"
    return sanitize_filename(f"{index:02d}_{base}")


def project_output_prefix(opju_path: Path, input_root: Path) -> str:
    rel = opju_path.relative_to(input_root)
    parts = [sanitize_filename(part) for part in rel.parts[:-1]]
    parts.append(sanitize_filename(opju_path.stem))
    return "__".join([part for part in parts if part])


def export_filename(project_prefix: str, workbook_prefix: str, workbook_count: int) -> str:
    if workbook_count == 1:
        return f"{project_prefix}.xlsx"
    return f"{project_prefix}__{workbook_prefix}.xlsx"


def existing_excel_exports(output_root: Path, filename_prefix: str) -> list[Path]:
    return sorted(
        [path for path in output_root.glob(f"{filename_prefix}*.xls*") if path.is_file()]
    )


def cleanup_existing_exports(output_root: Path, filename_prefix: str) -> None:
    for path in existing_excel_exports(output_root, filename_prefix):
        path.unlink()


def find_exported_excel(output_root: Path, filename_prefix: str) -> Path | None:
    matches = existing_excel_exports(output_root, filename_prefix)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def export_active_workbook(
    app,
    page,
    output_root: Path,
    export_name: str,
    overwrite: bool,
    cell_formats: bool,
    settle_seconds: float,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    file_prefix = Path(export_name).stem
    current_exports = existing_excel_exports(output_root, file_prefix)

    if current_exports and not overwrite:
        LOGGER.info("Skip existing export: %s", current_exports[0])
        return current_exports[0]

    cleanup_existing_exports(output_root, file_prefix)
    page.Activate()

    command = (
        f'expXLS name:="{lt_escape(file_prefix)}" '
        f'path:="{lt_escape(str(output_root))}" '
        f'overwrite:=1 sheets:=whole cellfmts:={1 if cell_formats else 0};'
    )
    ok = bool(app.Execute(command))
    if settle_seconds > 0:
        time.sleep(settle_seconds)

    exported = find_exported_excel(output_root, file_prefix)
    if ok and exported is not None:
        return exported

    raise RuntimeError(f"Origin expXLS failed for workbook export: {export_name}")


def convert_project(
    app,
    opju_path: Path,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
    cell_formats: bool,
    settle_seconds: float,
) -> list[Path]:
    app.NewProject()
    opened = bool(app.Load(str(opju_path)))
    if not opened:
        raise RuntimeError(f"Could not open project: {opju_path}")

    if settle_seconds > 0:
        time.sleep(settle_seconds)

    workbook_pages = app.WorksheetPages
    workbook_count = int(workbook_pages.Count)
    if workbook_count == 0:
        LOGGER.warning("No workbook pages found in %s", opju_path)
        return []

    project_prefix = project_output_prefix(opju_path, input_root)

    exported_files: list[Path] = []
    for index in range(workbook_count):
        workbook = workbook_pages(index)
        if workbook is None:
            LOGGER.warning("Skip null workbook page at index %d in %s", index, opju_path)
            continue
        workbook_prefix = workbook_export_prefix(workbook, index + 1)
        export_name = export_filename(project_prefix, workbook_prefix, workbook_count)
        exported = export_active_workbook(
            app,
            workbook,
            output_root,
            export_name,
            overwrite=overwrite,
            cell_formats=cell_formats,
            settle_seconds=settle_seconds,
        )
        exported_files.append(exported)
        LOGGER.info("  exported -> %s", exported)

    return exported_files


def main() -> int:
    args = parse_args()
    setup_logging()

    opju_files = sorted(args.input_root.rglob(args.pattern))
    if not opju_files:
        LOGGER.warning("No .opju files found under %s", args.input_root)
        return 0

    LOGGER.info("Found %d project files under %s", len(opju_files), args.input_root)

    try:
        win32 = import_win32com()
        app = dispatch_origin(win32, show_origin=args.show_origin)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 2

    failed_projects: list[tuple[Path, str]] = []
    exported_count = 0

    try:
        for index, opju_path in enumerate(opju_files, start=1):
            LOGGER.info("[%d/%d] %s", index, len(opju_files), opju_path)
            try:
                exported_files = convert_project(
                    app,
                    opju_path,
                    input_root=args.input_root,
                    output_root=args.output_root,
                    overwrite=args.overwrite,
                    cell_formats=args.cell_formats,
                    settle_seconds=args.settle_seconds,
                )
                exported_count += len(exported_files)
            except Exception as exc:
                failed_projects.append((opju_path, str(exc)))
                LOGGER.error("Failed: %s | %s", opju_path, exc)
    finally:
        try:
            app.Exit()
        except Exception:
            pass

    LOGGER.info("Finished. Exported %d Excel files.", exported_count)

    if failed_projects:
        LOGGER.error("Failed projects: %d", len(failed_projects))
        for path, message in failed_projects:
            LOGGER.error("  %s | %s", path, message)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
