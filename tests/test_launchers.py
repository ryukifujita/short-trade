"""起動ファイルのスタブ補完（launchers.ensure_launchers）の検査。"""
import os
from pathlib import Path

from short_trade.launchers import STUBS, ensure_launchers

REPO = Path(__file__).resolve().parents[1]


def test_creates_only_missing_stubs_and_marks_command_executable(tmp_path):
    (tmp_path / "run_mac.command").write_text("custom\n")
    created = ensure_launchers(tmp_path)
    assert "run_mac.command" not in created                      # 既存は触らない
    assert (tmp_path / "run_mac.command").read_text() == "custom\n"
    assert set(created) == set(STUBS) - {"run_mac.command"}
    mode = (tmp_path / "expand_mac.command").stat().st_mode
    assert mode & 0o111
    assert ensure_launchers(tmp_path) == []                       # 2回目は何もしない


def test_stub_templates_match_repository_files():
    """テンプレートとリポジトリの起動ファイルが食い違うと、補完で古い中身が作られる。"""
    for name, body in STUBS.items():
        assert (REPO / name).read_text(encoding="utf-8") == body, name
