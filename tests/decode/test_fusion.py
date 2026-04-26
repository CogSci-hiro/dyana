import numpy as np

from dyana.decode import fusion
from dyana.decode.params import DecodeTuningParams
from dyana.decode import state_space
from dyana.evidence.base import EvidenceTrack
from dyana.evidence.bundle import EvidenceBundle
from dyana.core.timebase import TimeBase


def test_fusion_module_has_docstring() -> None:
    assert fusion.__doc__ is not None


def test_pyannote_speech_penalizes_silence_when_other_speech_cues_exist() -> None:
    tb = TimeBase.canonical(n_frames=6)
    bundle = EvidenceBundle(timebase=tb)
    bundle.add_track("energy_smooth", EvidenceTrack("energy_smooth", tb, np.full(6, 0.8, dtype=np.float32), "score"))
    bundle.add_track("pyannote_speech", EvidenceTrack("pyannote_speech", tb, np.ones(6, dtype=np.float32), "probability"))

    base_scores = fusion.fuse_bundle_to_scores(bundle)
    recall_scores = fusion.fuse_bundle_to_scores(bundle, tuning_params=DecodeTuningParams.for_profile("recall-first"))

    sil_index = state_space.state_index("SIL")
    a_index = state_space.state_index("A")
    assert recall_scores[0, sil_index] < base_scores[0, sil_index]
    assert recall_scores[0, a_index] > base_scores[0, a_index]
