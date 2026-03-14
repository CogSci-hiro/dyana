from __future__ import annotations

from pathlib import Path

import numpy as np

from dyana.api.types import Alignment, AnnotationResult, ConversationalState, Diagnostics, IPU, Phoneme, Transcript, Word
from dyana.core.contracts.evidence import EvidenceBundle, EvidenceTrack
from dyana.io.artifacts import (
    load_alignment,
    load_diagnostics,
    load_evidence,
    load_ipus,
    load_states,
    load_transcript,
    save_alignment,
    save_diagnostics,
    save_evidence,
    save_ipus,
    save_states,
    save_transcript,
    write_run_artifacts,
)


def test_transcript_roundtrip(tmp_path: Path) -> None:
    transcript = Transcript(
        language="en",
        words=[
            Word(text="world", start=1.0, end=1.4, speaker="B", confidence=0.91),
            Word(text="hello", start=0.5, end=0.9, speaker="A", confidence=0.94),
        ],
    )

    path = tmp_path / "transcript.json"
    save_transcript(transcript, path)
    loaded = load_transcript(path)

    assert [word.text for word in loaded.words] == ["hello", "world"]
    assert loaded.language == "en"
    assert loaded.words[0].confidence == 0.94


def test_alignment_ipus_states_and_diagnostics_roundtrip(tmp_path: Path) -> None:
    alignment = Alignment(
        words=[Word(text="hello", start=0.5, end=0.9, speaker="A", confidence=0.94)],
        phonemes=[Phoneme(symbol="HH", start=0.5, end=0.6)],
    )
    ipus = [IPU(speaker="A", start=0.5, end=1.7)]
    states = [ConversationalState(label="A", start=0.5, end=1.7)]
    diagnostics = Diagnostics(
        metrics={"switch_rate": 0.83, "micro_ipu_rate": 0.12},
        flags=["rapid_alternation"],
    )

    alignment_path = tmp_path / "alignment.json"
    ipus_path = tmp_path / "ipus.json"
    states_path = tmp_path / "states.json"
    diagnostics_path = tmp_path / "diagnostics.json"

    save_alignment(alignment, alignment_path)
    save_ipus(ipus, ipus_path)
    save_states(states, states_path)
    save_diagnostics(diagnostics, diagnostics_path)

    assert load_alignment(alignment_path) == alignment
    assert load_ipus(ipus_path) == ipus
    assert load_states(states_path) == states
    assert load_diagnostics(diagnostics_path) == diagnostics


def test_evidence_npz_roundtrip(tmp_path: Path) -> None:
    tracks = {
        "energy": EvidenceTrack("energy", np.array([0.1, 0.2], dtype=np.float32), 0.01, 0.0),
        "vad": EvidenceTrack("vad", np.array([0.3, 0.4], dtype=np.float32), 0.01, 0.0),
        "voicing": EvidenceTrack("voicing", np.array([0.5, 0.6], dtype=np.float32), 0.01, 0.0),
        "speaker_a": EvidenceTrack("speaker_a", np.array([0.7, 0.8], dtype=np.float32), 0.01, 0.0),
        "speaker_b": EvidenceTrack("speaker_b", np.array([0.2, 0.1], dtype=np.float32), 0.01, 0.0),
        "overlap": EvidenceTrack("overlap", np.array([0.9, 0.0], dtype=np.float32), 0.01, 0.0),
    }
    bundle = EvidenceBundle(tracks=tracks, duration=0.02)

    path = tmp_path / "evidence.npz"
    save_evidence(bundle, path)
    loaded = load_evidence(path)

    assert loaded.duration == 0.02
    assert loaded.tracks.keys() == tracks.keys()
    assert np.allclose(loaded.tracks["energy"].values, tracks["energy"].values)
    with np.load(path, allow_pickle=False) as archive:
        assert float(archive["frame_hop"]) == 0.01
        assert float(archive["duration"]) == 0.02


def test_write_run_artifacts_creates_expected_files(tmp_path: Path) -> None:
    result = AnnotationResult(
        transcript=Transcript(words=[Word(text="hello", start=0.0, end=0.2, speaker="A", confidence=0.9)], language="en"),
        alignment=Alignment(words=[Word(text="hello", start=0.0, end=0.2, speaker="A", confidence=0.9)], phonemes=[Phoneme(symbol="HH", start=0.0, end=0.1)]),
        ipus=[IPU(speaker="A", start=0.0, end=0.5)],
        states=[ConversationalState(label="A", start=0.0, end=0.5)],
        diagnostics=Diagnostics(metrics={"overlap_ratio": 0.19}, flags=["rapid_alternation"]),
    )
    run_dir = tmp_path / "artifacts" / "runs" / "2026-03-14T10-15"

    write_run_artifacts(
        annotation_result=result,
        run_dir=run_dir,
        audio_path="/tmp/sample.wav",
        duration=0.5,
        pipeline_steps=["transcription", "alignment", "evidence", "decode"],
        run_id="2026-03-14T10-15",
        timestamp="2026-03-14T10:15:00+00:00",
    )

    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "transcript.json").exists()
    assert (run_dir / "alignment.json").exists()
    assert (run_dir / "ipus.json").exists()
    assert (run_dir / "states.json").exists()
    assert (run_dir / "diagnostics.json").exists()
    assert (run_dir / "alignment.TextGrid").exists()
