# DYANA — Week 2 Detailed Plan & Checklist (Option B primary, Option A optional)

**Week:** 2  
**Focus:** Robustness to **overlap + leakage** (cross-contamination)  
**Strategy:** **Option B (stereo geometry/heuristics) is primary.** Option A (separation-based evidence) is a stretch/optional add-on if time permits.

Week 2 must improve metrics on **hard segments** without breaking the Week 1 backbone.

---

## Week 2 definition of success

By end of Week 2:

- Leakage/contamination no longer spawns large numbers of micro-IPUs
- Overlap-heavy segments remain structurally stable (no speaker “ping-pong”)
- Easy-segment metrics do not regress
- All improvements are reflected in the Week 1 evaluation scorecard

---

## 1. Option B (primary) — Leakage & overlap evidence from stereo / multi-mic geometry

### 1.1 Inputs & assumptions

- Stereo or multi-channel audio is available (e.g., near–far or per-speaker mics)
- Channel asymmetry contains information about proximity / leakage
- We will compute evidence at canonical **10 ms** grid (resample as needed)

### 1.2 What Option B should produce (EvidenceTracks)

Minimum new tracks:

- `leakage_likelihood[t] ∈ [0, 1]`  
  Semantics: “speech-like activity that should not initiate or extend a turn”
- (Optional) `overlap_likelihood[t] ∈ [0, 1]`  
  Semantics: “probability both speakers are intentionally active”

Optional diagnostics (useful for debugging/calibration):
- `energy_ratio[t]` (channel A dominance)
- `xchan_similarity[t]` (cross-channel spectral similarity)
- `xchan_delay[t]` (estimated lag / echo signature)

### 1.3 Concrete stereo evidence recipes (keep minimal)

Pick **one minimal recipe** first; add others only if metrics demand.

#### Recipe B1 — Energy asymmetry + spectral similarity (recommended)
Compute per-frame:
- Energy per channel: `E_L(t), E_R(t)`
- Dominance ratio:
  - `dom(t) = E_L(t) / (E_L(t) + E_R(t) + eps)` (or choose the “near” channel if known)
- Cross-channel similarity (log-mel or MFCC cosine similarity):
  - `sim(t) = cos(φ_L(t), φ_R(t))`

Heuristic mapping:
- leakage rises when:
  - energy is low on a channel *but* similarity is high
  - dominance is extreme *and* the weaker channel still looks speech-like

#### Recipe B2 — Delayed correlation (room pickup cue)
Compute a short-lag cross-correlation between channels:
- stable non-zero lag → far-field pickup → leakage prior

Use as a soft boost to `leakage_likelihood`.

#### Recipe B3 — Envelope smear / decay (echo tail cue)
Use energy envelope slope/decay:
- slow decay + speech-like spectrum → leakage prior (esp. around turn offsets)

### 1.4 Failure modes (so you know what to expect)

- True overlap can create high similarity (don’t over-trust similarity alone)
- Backchannels may look leakage-like (short + low energy)
- If both mics are equally close, geometry cues weaken

Mitigation is via Axis 2 constraints, not “perfect evidence”.

---

## 2. Axis 2 (Week 2) — Make LEAK and OVL operational

Week 1 created the states. Week 2 makes them **behave correctly** under messy audio.

### 2.1 LEAK rules (must hold)

- **LEAK cannot start A or B** (very high cost / forbidden transition)
- LEAK should be cheap to enter/exit
- LEAK should *not* extend IPUs (tends to pull toward SIL rather than A/B)

Practical check:
- IPUs starting immediately after LEAK should be near-zero except with strong speech evidence.

### 2.2 Overlap rules (minimal, robust)

- Allow: `A → OVL → A` and `B → OVL → B`
- Allow `A → OVL → B` only with strong evidence (avoid forced turn switches)
- OVL should not force IPU splitting by default:
  - overlap annotation is a label; structural segmentation remains stable

### 2.3 Transition & duration tuning (metric-driven)

Tune:
- speaker-switch penalty
- OVL entry/exit penalties
- LEAK entry bias (via evidence fusion weights)

**Use the Week 1 scorecard** to validate every change.

---

## 3. Evaluation expectations in Week 2

Week 2 should primarily improve **hard cases**.

### 3.1 Metrics expected to improve

- **Micro-IPUs/min** ↓↓↓ (primary success metric)
- **Speaker switch rate/min** ↓↓
- **Boundary F1 @ 50 ms** ↑ on hard data
- Framewise IoU may stay similar (or slightly change)

### 3.2 Where to look first

- Synthetic leakage stress tests (controlled)
- Hard real segment (realistic)

Avoid over-weighting corpus averages early.

### 3.3 Add Week 2 diagnostics (recommended)

Log for each run:
- fraction of frames in LEAK
- fraction of frames in OVL
- number of IPUs that start within 100 ms of LEAK (should drop)
- distribution of IPU durations (median + tail)

---

## 4. Optional Option A (stretch) — Separation-based evidence (cached)

**Only attempt if Option B + decoder tuning are stable and metrics are moving.**

### 4.1 Goal (not clean separation)
Use separation as a **diagnostic** to improve leakage detection:
- residual energy ratio
- cross-stream similarity
- instability / flicker

Export the same EvidenceTracks:
- `leakage_likelihood[t]`
- optional `overlap_likelihood[t]`

### 4.2 Execution constraints
- Run once, cache outputs
- Do not put separation in any tight loop
- Start with a short segment benchmark to assess runtime on your GPU

---

## 5. End-of-Week-2 checklist (definition of done)

Tick **all** boxes before moving to Week 3.

### Option B evidence (primary path)
- [ ] At least one stereo-based leakage recipe implemented (B1 recommended)
- [ ] `leakage_likelihood` track exported on 10 ms grid
- [ ] (Optional) `overlap_likelihood` track exported
- [ ] Evidence is cached and rerunnable deterministically

### Decoder behavior (Axis 2)
- [ ] LEAK transitions prevent IPU initiation (validated in outputs)
- [ ] Overlap does not explode into speaker ping-pong
- [ ] Speaker-switch penalty tuned using metrics (not intuition only)

### Evaluation (must show progress)
- [ ] Hard real segment: micro-IPUs/min decreases substantially vs Week 1 baseline
- [ ] Hard real segment: speaker switch rate/min decreases
- [ ] Easy segment: no major regression in boundary F1
- [ ] Synthetic leakage test: robustness improves as leakage increases (graceful degradation)

### Diagnostics & interpretability
- [ ] Run report includes %LEAK and %OVL frames
- [ ] Can explain at least 2–3 key improvements by referencing evidence tracks + constraints
- [ ] No “mystery improvements” (if you can’t explain it, simplify)

### Optional (only if time allows)
- [ ] Separation-based leakage evidence benchmarked on your GPU
- [ ] Separation outputs cached
- [ ] Option A improves at least one hard-case metric without harming easy cases

---

## Bottom line

Week 2 is successful when:

- leakage becomes a **contained category** rather than a cascading failure
- overlap becomes a **label** rather than a segmentation catastrophe
- improvements are visible in the Week 1 scorecard on hard data

If you reach that point, Week 3 iteration becomes straightforward and safe.
