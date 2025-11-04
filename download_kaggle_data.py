#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下載 Kaggle 競賽資料並解壓縮的小工具。

預設會下載 `china-real-estate-demand-prediction` 競賽的資料，
存放在 `data/raw` 目錄並且解壓後刪除 zip 檔。
同時會將 `sample_submission.csv`、`test.csv` 提升到執行目錄，
以符合既有專案結構；可透過參數調整或停用。

使用方式：
    py -3 download_kaggle_data.py --output-dir train --force


若希望自訂輸出位置或保留壓縮檔，可使用指令列參數：
    python download_kaggle_data.py --output-dir train --keep-zip

執行前請先安裝 `kaggle` 套件並完成 API token 設定。
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
import shutil
from zipfile import ZipFile, BadZipFile

DEFAULT_COMPETITION = "china-real-estate-demand-prediction"


def run_kaggle_download(competition: str, output_dir: Path, force: bool) -> None:
    """呼叫 Kaggle CLI 下載競賽資料。"""
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(output_dir),
    ]
    if force:
        command.append("--force")
    logging.info("執行下載指令: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        logging.error(
            "找不到 kaggle 指令，請先安裝 kaggle 套件並設定 API token。\n"
            "安裝範例: pip install kaggle"
        )
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as exc:
        logging.error("kaggle 下載失敗 (exit code=%s)。", exc.returncode)
        raise SystemExit(exc.returncode) from exc


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    """解壓縮指定 zip 檔到 target_dir。"""
    logging.info("解壓縮 %s -> %s", zip_path.name, target_dir)
    try:
        with ZipFile(zip_path, "r") as archive:
            archive.extractall(target_dir)
    except BadZipFile as exc:
        logging.error("解壓縮失敗，zip 檔案可能毀損: %s", zip_path)
        raise SystemExit(1) from exc


def flatten_single_subdir(target_dir: Path, ignore: set[str] | None = None) -> None:
    """若 target_dir 內僅有一層資料夾，將其內容提升到 target_dir。"""
    ignore = ignore or set()
    dirs = [item for item in target_dir.iterdir() if item.is_dir()]
    files = [
        item for item in target_dir.iterdir() if item.is_file() and item.name not in ignore
    ]
    if len(dirs) != 1 or files:
        return
    nested = dirs[0]
    logging.info("展開巢狀資料夾: %s -> %s", nested, target_dir)
    for item in nested.iterdir():
        destination = target_dir / item.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        item.rename(destination)
    nested.rmdir()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下載 Kaggle 競賽資料集並解壓縮。")
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help="Kaggle 競賽代號 (預設: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        type=Path,
        help="下載與解壓縮後的儲存位置 (預設: %(default)s)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="保留下載的 zip 檔案 (預設為解壓後刪除)。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若目標目錄已存在相同檔案，仍強制重新下載。",
    )
    parser.add_argument(
        "--promote-files",
        nargs="*",
        default=["sample_submission.csv", "test.csv"],
        help=(
            "解壓後要移到 `--promote-dest` 的檔案清單。"
            "若想停用此功能，可透過 --no-promote。"
        ),
    )
    parser.add_argument(
        "--promote-dest",
        default=".",
        type=Path,
        help="提升檔案的目標資料夾 (預設: 執行指令時的目前路徑)。",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="不要移動任何檔案到其他資料夾。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_kaggle_download(args.competition, output_dir, args.force)

    zip_path = output_dir / f"{args.competition}.zip"
    if not zip_path.exists():
        logging.error("找不到下載得到的壓縮檔: %s", zip_path)
        return 1

    extract_zip(zip_path, output_dir)

    zip_exists = zip_path.exists()
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        logging.info("已刪除 zip 檔: %s", zip_path.name)
        zip_exists = False

    ignore_names = {"sample_submission.csv", "test.csv"}
    if zip_exists:
        ignore_names.add(zip_path.name)
    flatten_single_subdir(output_dir, ignore=ignore_names)

    promote_files = [] if args.no_promote else [f for f in (args.promote_files or []) if f]
    if promote_files:
        dest_dir = args.promote_dest.resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        for filename in promote_files:
            source = output_dir / filename
            if not source.exists():
                logging.warning("找不到要移動的檔案: %s", source)
                continue
            target = dest_dir / filename
            logging.info("移動 %s -> %s", source, target)
            target.write_bytes(source.read_bytes())
            source.unlink(missing_ok=True)

    logging.info("資料集下載完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
