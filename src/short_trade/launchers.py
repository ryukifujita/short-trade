"""起動ファイル（ダブルクリック用スタブ）を保証する。

起動ファイルは中身が2〜4行のスタブで、本体は scripts/ にある。自己更新（scripts/common.sh）は
「更新対象の一覧」に基づいて入れ替えるため、**一覧に無い新しい起動ファイルは初回の更新で降りてこない**
（一覧そのものが更新後にしか読まれない）。利用者に「もう一度開いて」と言わずに済むよう、
CLI の起動時に足りないスタブをここから作る。既にあるものは触らない。
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STUBS: dict[str, str] = {
    "start_mac.command": (
        "#!/bin/bash\n"
        "# 初回セットアップ（macOS）。中身は scripts/setup.sh。このファイルは変わらない。\n"
        'cd "$(dirname "$0")" && exec bash scripts/setup.sh\n'
    ),
    "run_mac.command": (
        "#!/bin/bash\n"
        "# 2回目以降（macOS）。中身は scripts/run.sh（起動時に自動で最新化）。このファイルは変わらない。\n"
        'cd "$(dirname "$0")" && exec bash scripts/run.sh\n'
    ),
    "expand_mac.command": (
        "#!/bin/bash\n"
        "# ユニバース拡張（macOS、1回だけ）。中身は scripts/expand.sh。このファイルは変わらない。\n"
        'cd "$(dirname "$0")" && exec bash scripts/expand.sh\n'
    ),
    "tune_mac.command": (
        "#!/bin/bash\n"
        "# パラメータ探索（macOS）。中身は scripts/tune.sh。このファイルは変わらない。\n"
        'cd "$(dirname "$0")" && exec bash scripts/tune.sh\n'
    ),
    "start_windows.bat": (
        "@echo off\n"
        "rem 初回セットアップ（Windows）。中身は scripts\\setup.bat。このファイルは変わらない。\n"
        'cd /d "%~dp0"\n'
        'if exist "scripts.new" (rmdir /s /q scripts & move /Y scripts.new scripts >nul)\n'
        "call scripts\\setup.bat\n"
    ),
    "run_windows.bat": (
        "@echo off\n"
        "rem 2回目以降（Windows）。中身は scripts\\run.bat（起動時に自動で最新化）。このファイルは変わらない。\n"
        'cd /d "%~dp0"\n'
        'if exist "scripts.new" (rmdir /s /q scripts & move /Y scripts.new scripts >nul)\n'
        'if exist "run_windows.new.bat" del /q "run_windows.new.bat"\n'
        "call scripts\\run.bat\n"
    ),
    "expand_windows.bat": (
        "@echo off\n"
        "rem ユニバース拡張（Windows、1回だけ）。中身は scripts\\expand.bat。このファイルは変わらない。\n"
        'cd /d "%~dp0"\n'
        'if exist "scripts.new" (rmdir /s /q scripts & move /Y scripts.new scripts >nul)\n'
        "call scripts\\expand.bat\n"
    ),
    "tune_windows.bat": (
        "@echo off\n"
        "rem パラメータ探索（Windows）。中身は scripts\\tune.bat。このファイルは変わらない。\n"
        'cd /d "%~dp0"\n'
        'if exist "scripts.new" (rmdir /s /q scripts & move /Y scripts.new scripts >nul)\n'
        "call scripts\\tune.bat\n"
    ),
}


def ensure_launchers(root: Path = ROOT) -> list[str]:
    """無い起動ファイルだけ作る。作ったファイル名を返す。"""
    created: list[str] = []
    for name, body in STUBS.items():
        path = root / name
        if path.exists():
            continue
        try:
            path.write_bytes(body.encode("utf-8"))
            if name.endswith(".command"):
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            created.append(name)
        except OSError:
            continue
    return created
