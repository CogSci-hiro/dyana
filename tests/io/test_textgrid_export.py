from pathlib import Path

from dyana.decode.ipu import Segment
from dyana.io.praat_textgrid import write_textgrid


def test_textgrid_written(tmp_path: Path) -> None:
    segments = [Segment(0.0, 1.0, "A")]
    path = tmp_path / "out.TextGrid"
    write_textgrid(path, speaker_a=segments, speaker_b=[], overlap=[], leak=[])
    content = path.read_text()
    assert "SpeakerA" in content
    assert "IntervalTier" in content
