# DyANA evaluation runs

## Manifest template (3 tiers)

```json
[
  {
    "id": "syn_leak_1",
    "tier": "synthetic",
    "scenario": "leakage_stress",
    "audio_path": null,
    "ref_path": null
  },
  {
    "id": "easy_1",
    "tier": "easy",
    "audio_path": "/abs/path/easy.wav",
    "ref_path": "/abs/path/easy.TextGrid"
  },
  {
    "id": "hard_1",
    "tier": "hard",
    "audio_path": "/abs/path/hard.wav",
    "ref_path": "/abs/path/hard.TextGrid"
  }
]
```

## Commands

```bash
dyana eval --manifest manifest.json --out-dir out --run-name baseline
dyana tune --manifest manifest.json --baseline out/baseline/scorecard.json --out-dir out --run-name current
```

## Outputs

- `out/baseline/scorecard.json`
- `out/baseline/scorecard.csv`
- `out/current/scorecard.json`
- `out/current/scorecard.csv`
- `out/current/delta.json`
- `out/current/delta.csv`
