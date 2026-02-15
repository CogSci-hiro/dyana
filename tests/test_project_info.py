from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

import project_info


def test_human_bytes_formats_reasonably() -> None:
    assert project_info._human_bytes(0) == "0.0 B"
    assert project_info._human_bytes(1023) == "1023.0 B"
    assert project_info._human_bytes(1024) == "1.0 KB"
    assert project_info._human_bytes(1024 * 1024) == "1.0 MB"


def test_safe_rel_falls_back_to_absolute(tmp_path: Path) -> None:
    root = tmp_path / "root"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    p = other / "x.txt"
    p.write_text("x", encoding="utf-8")
    assert project_info._safe_rel(p, root) == str(p)


def test_is_probably_binary_detects_nul(tmp_path: Path) -> None:
    p = tmp_path / "bin.dat"
    p.write_bytes(b"\x00abc")
    assert project_info._is_probably_binary(p) is True

    t = tmp_path / "text.txt"
    t.write_text("hello", encoding="utf-8")
    assert project_info._is_probably_binary(t) is False


def test_load_yaml_requires_mapping(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        project_info._load_yaml(p)


def test_main_writes_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("hi", encoding="utf-8")
    (root / "config.yaml").write_text("io:\n  out_dir: out\n", encoding="utf-8")

    out_dir = root / "out"
    out_dir.mkdir()
    (out_dir / "artifact.txt").write_text("a", encoding="utf-8")

    def fake_run(cmd: list[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
        if cmd[:2] == ["git", "rev-parse"]:
            return 0, "deadbeef", ""
        if cmd[:2] == ["git", "status"]:
            return 0, "", ""
        if cmd[:2] == ["pip", "freeze"]:
            return 0, "pytest==0\n", ""
        return 0, "", ""

    monkeypatch.setattr(project_info, "_run", fake_run)
    monkeypatch.chdir(root)

    out_path = root / "snap.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["project_info.py", "--out", str(out_path), "--config", str(root / "config.yaml")],
    )
    project_info.main()

    written = out_path.read_text(encoding="utf-8")
    assert "# Project Snapshot" in written
    assert "## Repository File Summary" in written
    assert "## Output Artifact Inventory" in written

