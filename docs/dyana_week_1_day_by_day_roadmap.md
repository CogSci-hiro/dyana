# DYANA — Week 1 Day-by-Day Roadmap (Checklist)

✅ done 
⌛running/slow 
🚫paused 
⏱️needs review 
🔄in progress

**Start date:** Monday **2026-02-09**  
**End date:** Friday **2026-02-13**  
**Goal:** Build the *spine* of DYANA — a deterministic, end-to-end pipeline from audio to IPUs with metrics.

> Rule of the week: **no learning, no ASR, no iteration loops**. If it’s not measurable, it doesn’t belong here.

---

## Day 1 — Monday (2026-02-09)
### Canonical timebase & core contracts

**Objective:** Lock the global temporal backbone and data contracts. Everything depends on this.

### Concrete goals
- Define a single canonical time grid (10 ms hop)
- Implement resampling utilities
- Define and validate evidence data structures

### Checklist
- [✅] Define `TimeBase` (frame ↔ time mapping)
- [✅] Enforce 10 ms global hop (`t = frame_idx * 0.01`)
- [✅] Implement upsampling (e.g. 20 ms → 10 ms)
- [✅] Implement downsampling with explicit aggregation (max / mean)
- [✅] Implement `EvidenceTrack`
  - [✅] shape validation
  - [✅] explicit semantics (probability / score / logit)
  - [✅] optional confidence
- [✅] Implement `EvidenceBundle`
  - [✅] order-independent
  - [✅] missing tracks allowed
- [✅] Serialization to disk (NPZ / JSON)

**Exit criteria:**
- Any evidence track can be resampled onto the canonical grid
- Shape or timebase mismatch fails loudly

---

## Day 2 — Tuesday (2026-02-10)
### Axis 2 decoder skeleton (no real audio)

**Objective:** Build a stable structured decoder *before* touching real evidence.

### Concrete goals
- Implement minimal state space
- Implement Viterbi-style decoding
- Encode conversational constraints

### Checklist
- [✅] Define states: `SIL`, `A`, `B`, `OVL`, `LEAK`
- [✅] Implement framewise state score interface
- [✅] Implement Viterbi-style DP
- [✅] Add transition penalties:
  - [✅] state persistence cheap
  - [✅] speaker switch expensive
  - [✅] minimum IPU duration
  - [✅] minimum silence duration
- [✅] Enforce: **LEAK cannot initiate IPUs**
- [✅] Run decoder using fake / random evidence

**Exit criteria:**
- Decoder runs deterministically
- No A↔B ping-pong
- No frame-level jitter

---

## Day 3 — Wednesday (2026-02-11)
### Axis 1: cheap, deterministic evidence

**Objective:** Plug in minimal, reliable evidence sources as independent modules.

### Concrete goals
- Extract trivial but robust speech cues
- Export all cues as independent EvidenceTracks

### Checklist
- [✅] Energy / RMS envelope per frame
- [✅] Smoothed energy envelope (50–100 ms)
- [✅] WebRTC VAD integration
  - [✅] exported as *soft* likelihood
  - [✅] never binary
- [✅] Minimal prosodic cues:
  - [✅] voiced / unvoiced flag
  - [✅] energy slope
- [✅] Each cue exported as its own EvidenceTrack
- [✅] Evidence caching enabled

**Exit criteria:**
- Removing any single evidence track does not crash decoding
- Evidence artifacts are inspectable on disk

---

## Day 4 — Thursday (2026-02-12)
### End-to-end pipeline & outputs

**Objective:** Make DYANA actually run from audio to annotation.

### Concrete goals
- Wire all components together
- Produce human-readable outputs

### Checklist
- [✅] Audio loading (mono + multi-channel)
- [✅] End-to-end execution:
  ```
  audio → evidence bundle → decoder → state sequence
  ```
- [✅] IPU extraction from decoded states
- [✅] Praat TextGrid export:
  - [✅] Speaker A IPUs
  - [✅] Speaker B IPUs
  - [✅] Overlap
  - [✅] Leakage
- [✅] Dump intermediate artifacts (evidence + decoder outputs)

**Exit criteria:**
- Pipeline runs on multiple files without crashing
- Outputs are interpretable by inspection
- Results are identical across repeated runs

---

## Day 5 — Friday (2026-02-13)
### Evaluation harness & baseline scorecard

**Objective:** Ensure every future improvement is measurable.

### Concrete goals
- Implement core metrics
- Run evaluation on multiple dataset tiers

### Checklist
- [✅] Boundary F1 implementation
  - [✅] ±20 ms tolerance
  - [✅] ±50 ms tolerance
- [✅] Framewise IoU (speech vs silence, per speaker)
- [✅] Structural metrics:
  - [✅] micro-IPUs / min
  - [✅] speaker switches / min
  - [✅] rapid alternations
- [ ] Evaluation on:
  - [ ] synthetic / semi-synthetic data
  - [ ] easy real segment
  - [ ] hard real segment
- [ ] Baseline scorecard written to disk

**Exit criteria:**
- Metrics rerun identically across runs
- Removing evidence changes scores *predictably*

---

## End-of-Week Definition of Done

You may proceed to Week 2 **only if all are true**:

- [ ] Single canonical 10 ms timebase everywhere
- [ ] EvidenceTrack / EvidenceBundle used end-to-end
- [ ] Decoder stable with fake or missing evidence
- [ ] LEAK state blocks spurious IPUs
- [ ] TextGrid outputs look sane
- [ ] Metrics exist and are reproducible

> If any box is unchecked: fix it before adding complexity.

