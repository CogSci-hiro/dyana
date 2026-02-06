# DYANA v0 Roadmap — Minimal Functional Pipeline by End of Month

**Target date:** Feb 28, 2026  
**Project:** DYANA — DYadic Annotation of Naturalistic Audio

This roadmap is designed to deliver a **deterministic, end-to-end, minimally functional DYANA pipeline** within ~3 weeks.  
The emphasis is on **robust structure**, **cost control**, and **debuggability**, not on perfect models.

---

## Definition of success (v0)

By the end of the month, DYANA should be able to run:

```bash
dyana run --audio <file or dyad/run> --out <output_dir>
```

and produce:

- IPU annotations (Praat TextGrid) for Speaker A and B
- Overlap / leakage annotations
- A JSON quality report with per-IPU confidence and flags
- Cached intermediate artifacts (evidence tracks, separation outputs)
- Deterministic, reproducible output (fixed seeds)

No Whisper-in-the-loop. No training required.

---

## Week 1 (Feb 6–Feb 13): Build the spine

**Goal:** A complete end-to-end pipeline that runs, even with crude evidence.

### Deliverables

#### 1. Core infrastructure
- Canonical time grid (recommended: **10 ms**)
- `EvidenceTrack` and `EvidenceBundle` contracts
- Evidence resampling utilities (20 ms → 10 ms)
- Deterministic caching layer (hash-based)

#### 2. Axis 2 — Minimal structured decoder
- State space:
  - silence
  - speaker A
  - speaker B
  - overlap
  - leakage (can be weak initially)
- Simple constrained decoder (Viterbi-style acceptable for v0)
- Constraints:
  - persistence / hysteresis
  - minimum IPU duration
  - minimum silence duration
  - speaker-switch penalty

#### 3. Axis 1 — Starter evidence modules (cheap)
- Energy-based / WebRTC VAD
- Basic prosody cues (energy envelope, voiced/unvoiced)
- Mic-based priors if multi-mic data is available

#### 4. IO + CLI
- WAV loading (mono + multi-channel)
- Praat TextGrid export
- Artifact dumping (JSON / NPZ)

### Acceptance criteria
- Pipeline runs on 2–3 recordings without crashing
- IPUs roughly align with audible speech
- No extreme frame-level jitter

---

## Week 2 (Feb 14–Feb 20): Handle overlap & leakage

**Goal:** Address the main failure mode — cross-contamination destroying IPUs.

### Deliverables

#### 1. Axis 1 — Overlap / leakage evidence
Choose one:
- Off-the-shelf separation (e.g., SepFormer) → derive overlap/leakage likelihoods  
or
- Heuristic proxy:
  - cross-channel similarity
  - delayed energy correlation
  - far-field spectral signatures

#### 2. Axis 2 — Leakage-aware decoding
- Leakage cannot start IPUs
- Overlap can exist without forcing turn splits
- Backchannels do not split IPUs

#### 3. Quality & uncertainty diagnostics
- Boundary instability metric
- Posterior entropy per region
- Invariant checks:
  - rapid alternation
  - too-short IPUs
  - extreme speaking-time imbalance

### Acceptance criteria
- Fewer micro-IPUs in overlap-heavy segments
- Overlap flagged instead of misattributed
- Clear diagnostics for ambiguous regions

---

## Week 3 (Feb 21–Feb 28): Iteration & polish

**Goal:** Add a lightweight EM-like loop and stabilize outputs.

### Deliverables

#### 1. Axis 3 — Iterative refinement (v0)
- Iteration 0: full evidence → decode
- Diagnostics:
  - instability
  - invariant violations
- Iteration 1–2:
  - adjust evidence weights
  - adjust decoder penalties
  - selectively recompute unstable regions only

#### 2. Caching policy
- Cache SSL embeddings (if used)
- Cache separation outputs
- Never loop expensive ASR

#### 3. Evaluation harness
- Boundary F1 with tolerance (20 ms / 50 ms)
- Speaker consistency metrics
- Small gold set (15–30 min) for regression tests

#### 4. Documentation
- How to run DYANA
- Meaning of outputs
- Known failure modes + manual queue guidance

### Acceptance criteria
- Stable convergence in ≤3 iterations
- Deterministic results across runs
- Full mini-dataset runs overnight without intervention

---

## Explicit non-goals for v0 (to protect the deadline)

- No SepFormer training or fine-tuning
- No end-to-end neural models
- No full syntactic / pragmatic modeling
- No Whisper in the refinement loop

---

## Optional stretch goals (only if ahead)

- Pyannote diarization wrapper as optional Axis 1 module
- Cached wav2vec2 embeddings as structural evidence
- Selective Whisper for high-uncertainty segments (post-loop)

---

## Main risks & mitigations

### Risk: Decoder instability early on
**Mitigation:** Implement Axis 2 first with synthetic or dummy evidence.

### Risk: Leakage logic complexity
**Mitigation:** Treat leakage as a state, not something to eliminate.

### Risk: Unclear evaluation
**Mitigation:** Define boundary F1 and micro-IPU rate early.

---

## Bottom line

This roadmap prioritizes:
- Structural correctness over model sophistication
- Determinism and debuggability
- A system that can evolve, rather than a brittle demo

By Feb 28, DYANA v0 should be **usable, inspectable, and extensible**.
