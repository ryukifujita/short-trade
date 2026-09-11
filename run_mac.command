#!/bin/bash
# 2回目以降（macOS）。中身は scripts/run.sh（起動時に自動で最新化）。このファイルは変わらない。
cd "$(dirname "$0")" && exec bash scripts/run.sh
