from pathlib import Path

from dyana.decode.ipu import Segment
from dyana.io.praat_textgrid import parse_textgrid, write_textgrid


def test_textgrid_written(tmp_path: Path) -> None:
    segments = [Segment(0.0, 1.0, "A")]
    path = tmp_path / "out.TextGrid"
    write_textgrid(path, speaker_a=segments, speaker_b=[], overlap=[], leak=[])
    content = path.read_text()
    assert "SpeakerA" in content
    assert "IntervalTier" in content
    assert "item [1]:" in content
    assert "xmax = 1" in content


def test_textgrid_round_trips_segments(tmp_path: Path) -> None:
    speaker_a = [Segment(0.0, 1.0, "A")]
    overlap = [Segment(1.0, 1.5, "OVL")]
    path = tmp_path / "roundtrip.TextGrid"

    write_textgrid(path, speaker_a=speaker_a, speaker_b=[], overlap=overlap, leak=[])
    parsed = parse_textgrid(path)

    assert parsed["SpeakerA"] == speaker_a
    assert parsed["Overlap"] == overlap


def test_textgrid_fills_gaps_with_silence_symbol(tmp_path: Path) -> None:
    speaker_a = [Segment(0.25, 0.5, "A")]
    path = tmp_path / "silence_fill.TextGrid"

    write_textgrid(path, speaker_a=speaker_a, speaker_b=[], overlap=[], leak=[])
    content = path.read_text()

    assert 'text = "#"' in content
    assert "intervals: size = 2" in content
    parsed = parse_textgrid(path)
    assert parsed["SpeakerA"] == speaker_a
