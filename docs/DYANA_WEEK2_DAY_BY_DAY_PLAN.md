# DYANA --- Week 2 Detailed Execution Plan

**Week:** 2\
**Focus:** Robustness to overlap + leakage\
**Primary:** Option B (stereo geometry)\
**Stretch:** Option A (separation-based)

------------------------------------------------------------------------

## Day 1 --- Monday

### Implement minimal stereo leakage evidence (Recipe B1)

**Objective:** Export a deterministic `leakage_likelihood[t]` track on
the 10 ms grid.

### Concrete goals

-   Compute per-channel energy on canonical 10 ms grid
-   Implement dominance ratio
-   Implement cross-channel spectral similarity (cosine on log-mel or
    MFCC)
-   Define minimal heuristic mapping → `leakage_likelihood ∈ [0,1]`
-   Export as `EvidenceTrack`
-   Cache deterministically

### Checklist

-   [✅] Energy per channel computed on canonical grid
-   [✅] Dominance ratio implemented (stable w/ eps)
-   [✅] Cross-channel similarity implemented
-   [✅] Leakage mapping function implemented
-   [✅] EvidenceTrack semantics explicitly documented
-   [✅] Export validated against TimeBase
-   [✅] Evidence cached and reloadable
-   [✅] Removing track does not crash decoder

### Exit criteria

-   Evidence looks interpretable when plotted
-   Track is deterministic across runs
-   No decoder change yet

------------------------------------------------------------------------

## Day 2 --- Tuesday

### Integrate leakage into decoder (LEAK behavior enforcement)

**Objective:** Make LEAK operational, not decorative.

### Concrete goals

-   Inject `leakage_likelihood` into state scoring
-   Enforce:
    -   LEAK cannot initiate IPUs
    -   LEAK cheap to enter/exit
    -   LEAK biased toward SIL, not A/B
-   Add diagnostic counters

### Checklist

-   [ ] LEAK state scoring wired to evidence
-   [ ] Transition penalties adjusted
-   [ ] LEAK → A/B initiation strongly penalized
-   [ ] IPU start-after-LEAK counter logged
-   [ ] Synthetic leakage test run
-   [ ] No crashes with missing leakage track

### Exit criteria

-   Micro-IPUs drop on synthetic leakage test
-   Decoder behavior explainable in terms of evidence

------------------------------------------------------------------------

## Day 3 --- Wednesday

### Overlap stabilization (minimal OVL tuning)

**Objective:** Prevent overlap from creating speaker ping-pong.

### Concrete goals

-   Implement optional `overlap_likelihood`
-   Tune:
    -   A → OVL → A allowed
    -   B → OVL → B allowed
    -   A → OVL → B high penalty unless strong evidence
-   Ensure OVL does not force IPU splitting

### Checklist

-   [ ] OVL evidence implemented (can be simple proxy)
-   [ ] Transition penalties adjusted
-   [ ] OVL does not explode switch rate
-   [ ] Structural metrics evaluated on hard segment
-   [ ] Rapid alternation metric checked

### Exit criteria

-   Speaker-switch rate decreases on hard segment
-   No regression on easy segment

------------------------------------------------------------------------

## Day 4 --- Thursday

### Metric-driven tuning & stress testing

**Objective:** Improve hard cases without breaking easy ones.

### Concrete goals

-   Run full Week 1 scorecard
-   Compare:
    -   micro-IPUs/min
    -   switch rate/min
    -   Boundary F1 @20/50ms
-   Tune only:
    -   speaker-switch penalty
    -   LEAK entry bias
    -   OVL transition costs

### Checklist

-   [ ] Synthetic leakage test evaluated
-   [ ] Hard real segment evaluated
-   [ ] Easy real segment evaluated
-   [ ] Metric deltas logged vs Week 1 baseline
-   [ ] No unexplained improvements
-   [ ] No large easy-case regression

### Exit criteria

-   Hard segment: micro-IPUs ↓↓↓
-   Easy segment: boundary F1 stable
-   All improvements attributable to identifiable mechanism

------------------------------------------------------------------------

## Day 5 --- Friday

### Diagnostics, reporting, and containment validation

**Objective:** Make leakage a contained category.

### Concrete goals

-   Add run-level diagnostics:
    -   %LEAK frames
    -   %OVL frames
    -   IPUs starting within 100 ms of LEAK
    -   IPU duration distribution summary
-   Write brief interpretability report
-   Freeze Week 2 baseline scorecard

### Checklist

-   [ ] Diagnostics exported to JSON
-   [ ] Histogram of IPU durations computed
-   [ ] LEAK fraction logged
-   [ ] OVL fraction logged
-   [ ] Week 2 scorecard written to disk
-   [ ] Deterministic rerun validated

### Exit criteria

-   Leakage no longer cascades into micro-IPU explosion
-   Overlap behaves like a label, not segmentation chaos
-   Metrics reproducible

------------------------------------------------------------------------

# Optional Stretch Block

## Option A --- Separation-based leakage evidence (cached)

### Concrete goals

-   Run separation on short benchmark segment
-   Derive residual energy ratio
-   Export `leakage_likelihood` equivalent track
-   Compare vs Option B evidence

### Checklist

-   [ ] Separation runtime benchmarked
-   [ ] Outputs cached
-   [ ] Leakage evidence derived
-   [ ] At least one hard-case metric improves
-   [ ] No easy-case regression

### Abort rule

If runtime is unstable or improvement is marginal → drop for now.

------------------------------------------------------------------------

# Week 2 Definition of Done

-   [ ] Hard segment micro-IPUs/min substantially reduced
-   [ ] Hard segment switch rate reduced
-   [ ] Easy segment F1 stable
-   [ ] LEAK behaves predictably
-   [ ] Overlap does not create ping-pong
-   [ ] All improvements visible in scorecard
-   [ ] No mystery gains
