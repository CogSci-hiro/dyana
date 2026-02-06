# DYANA — Week 1 Detailed Plan & Checklist

**Week:** 1  
**Goal:** Establish a complete, minimal **audio → IPU output → evaluation** pipeline with deterministic behavior and baseline metrics.

This week is about building the **spine** of DYANA.  
If Week 1 is done correctly, every later improvement will show up as **measurable metric gains**, not just subjective “looks better”.

---

## Week 1 high-level objectives

By the end of Week 1, DYANA must support:

1. End-to-end execution:
   ```
   audio → evidence → structured decoding → IPUs → evaluation metrics
   ```
2. Deterministic, reproducible outputs
3. A baseline scorecard on curated test data

No training. No ASR. No iteration loop yet.

---

## 1. Core contracts & timebase

### 1.1 Canonical timebase

- Single global time grid: **10 ms frame hop**
- Frame index ↔ time mapping:
  ```
  t_seconds = frame_index * 0.01
  ```
- All modules must export to this grid
- Internal computation at other resolutions is allowed, but **resampling is mandatory at boundaries**

### 1.2 EvidenceTrack (conceptual contract)

Every Axis-1 module outputs one or more `EvidenceTrack`s with:

- `name`: semantic identifier (e.g. `"vad"`, `"speaker_A"`, `"overlap"`)
- `timebase`: reference to canonical grid
- `values`: array shaped `(T,)` or `(T, K)`
- `semantics`: what the values mean (probability, score, logit)
- `confidence` (optional): reliability estimate
- `metadata`: model name, parameters, provenance

Key rule:
> EvidenceTracks are **soft evidence**, never hard decisions.

### 1.3 EvidenceBundle

- Named collection of EvidenceTracks
- Order-independent
- Sparse allowed (modules may be missing)
- Cacheable as a unit

Axis 2 must operate robustly with partial bundles.

---

## 2. Axis 2 — Minimal structured decoder

### 2.1 Minimal state space (v0)

At each frame, exactly one state is active:

- `SIL` — silence / no speech
- `A` — speaker A holds the floor
- `B` — speaker B holds the floor
- `OVL` — intentional overlap (both active)
- `LEAK` — speech-like contamination with no conversational agency

**LEAK is critical**: it absorbs cross-talk, echo, and separation residue without spawning turns.

### 2.2 Decoding paradigm

- MAP sequence decoding over time
- HMM-like **Viterbi-style dynamic programming**
- Not a generative HMM
- No learning required

Objective (conceptually):
```
sum_t state_score(state_t | evidence_t)
+ sum_t transition_cost(state_{t-1} → state_t)
```

### 2.3 Core constraints

Implemented as transition penalties:

- State persistence is cheap
- Speaker switches are expensive
- Minimum IPU duration enforced (e.g. ≥200 ms)
- Minimum silence enforced (e.g. ≥100 ms)
- **LEAK cannot start IPUs**
- OVL does not force IPU splits

Result:
> No frame-level jitter, no ping-pong speaker flips.

---

## 3. Axis 1 — Starter evidence (cheap, deterministic)

### 3.1 WebRTC VAD

- Fast, CPU-based voice activity detector
- Frame-based (10/20/30 ms)
- Robust speech presence prior
- Outputs speech likelihood (or proxy)

Used only as **one evidence source**, not ground truth.

### 3.2 Energy & envelope features

- RMS / log-energy per frame
- Smoothed envelope (50–100 ms)

Used for:
- speech presence bias
- turn-end decay cues
- leakage vs near-mic discrimination

### 3.3 Prosodic cues (minimal)

Prosody here means **turn-shape cues**, not linguistics.

Extract:
- Voiced / unvoiced flag
- Rough F0 (if available)
- Energy slope

Used to compute:
- boundary likelihood (soft)
- continuation vs ending bias

Prosody **nudges**, never decides.

---

## 4. Evaluation framework (must exist in Week 1)

### 4.1 Evaluation datasets

Run evaluation on four tiers:

1. **Synthetic / semi-synthetic**
   - Clean speech + artificial leakage / overlap
   - Ground truth known by construction

2. **Easy real segment**
   - Clean turn-taking
   - Minimal overlap
   - 2–5 minutes

3. **Hard real segment**
   - Frequent overlap, backchannels, leakage
   - Ambiguous but realistic

4. **Full corpus (known ground truth)**
   - Used for regression and distributional sanity checks

### 4.2 Core metrics (v0)

#### Boundary F1
- Start/end boundaries
- Tolerance windows:
  - ±20 ms (strict)
  - ±50 ms (practical)

#### Framewise IoU / Jaccard
- Binary speech vs silence
- Computed per speaker on 10 ms grid

#### Structural metrics
- Micro-IPUs/min (<150–250 ms)
- Speaker switch rate/min
- Rapid alternation count (A↔B↔A <300 ms)

These metrics must be:
- deterministic
- cheap to compute
- logged to disk

### 4.3 Baseline scorecard

A single table summarizing all datasets, rerunnable every week.

---

## 5. End-of-Week-1 checklist (definition of done)

Tick **all** boxes before moving to Week 2.

### Pipeline & infrastructure
- [ ] Single canonical 10 ms timebase implemented
- [ ] EvidenceTrack and EvidenceBundle contracts defined and used
- [ ] Evidence resampling implemented and tested
- [ ] Deterministic caching in place

### Decoder (Axis 2)
- [ ] Minimal state space implemented (SIL, A, B, OVL, LEAK)
- [ ] Viterbi-style constrained decoding implemented
- [ ] Minimum IPU and silence durations enforced
- [ ] LEAK state blocks IPU initiation

### Evidence (Axis 1)
- [ ] WebRTC VAD integrated as evidence
- [ ] Energy / envelope features integrated
- [ ] Basic prosodic cues integrated

### Evaluation
- [ ] Evaluation code runs automatically after inference
- [ ] Boundary F1 @20ms and @50ms implemented
- [ ] Framewise IoU implemented
- [ ] Micro-IPU and switch-rate metrics implemented
- [ ] All four dataset tiers evaluated

### Sanity checks
- [ ] Running with fake / random evidence does not crash
- [ ] Removing an evidence track does not crash decoding
- [ ] Results are identical across runs with fixed seed
- [ ] Outputs are interpretable (can explain *why* a boundary exists)

---

## Bottom line

If every checkbox above is ticked:

- DYANA has a **stable backbone**
- Metrics provide a **baseline reference**
- Week 2 improvements will be **measurable and meaningful**

If any checkbox is missing:
> Fix it before adding complexity.

This discipline is what keeps DYANA from becoming an opaque research prototype.
