<p align="center">
  <a href="https://github.com/CogSci-hiro/dyana">
    <img src="assets/logo.svg" alt="DYANA logo" width="180">
  </a>
</p>

<h1 align="center">DYANA</h1>

<p align="center"><strong>DYadic Annotation of Naturalistic Audio</strong></p>

<p align="center">
  Deterministic, inspectable annotation pipelines for conversational audio.
</p>

<p align="center">
  <a href="https://cogsci-hiro.github.io/dyana/">GitHub Pages documentation</a>
  ·
  <a href="https://github.com/CogSci-hiro/dyana">Repository</a>
</p>

DYANA is a Python toolkit for analyzing dyadic audio with an emphasis on traceability. It extracts evidence from recordings, combines those signals into constrained decoding steps, and writes artifacts that make it possible to inspect what happened at each stage instead of treating the pipeline as a black box.

The repository is still evolving, but the end-to-end CLI is already usable for local WAV and FLAC inputs. Today, the most stable path is the `dyana run` workflow, which produces evidence tracks, decode outputs, diagnostics, and Praat-friendly exports.

## What DYANA is built for

- Deterministic audio annotation workflows
- Inspectable intermediate artifacts and diagnostics
- Constrained decoding of conversational state sequences
- Evaluation and tuning loops for improving pipeline behavior
- Export paths that fit downstream analysis and annotation workflows

## Documentation

The project documentation is published on GitHub Pages:

- [DYANA docs](https://cogsci-hiro.github.io/dyana/)

Useful entry points:

- [Quickstart](https://cogsci-hiro.github.io/dyana/quickstart.html)
- [CLI reference](https://cogsci-hiro.github.io/dyana/cli.html)
- [Architecture overview](https://cogsci-hiro.github.io/dyana/architecture.html)
- [API reference](https://cogsci-hiro.github.io/dyana/api/index.html)

## Installation

Create and activate a virtual environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

- Tests: `pip install -e '.[test]'`
- Development tooling and docs: `pip install -e '.[dev,docs]'`
- Whisper ASR support: `pip install -e '.[asr]'`

## Quickstart

Run the default end-to-end pipeline on a single file:

```bash
dyana run path/to/sample.wav --out-dir out
```

Or run it over a directory of `.wav` and `.flac` files:

```bash
dyana run path/to/audio_dir --out-dir out
```

Typical outputs look like this:

```text
out/
  sample/
    sample.TextGrid
    decode/
    evidence/
    logs/
```

For fail-fast debugging with a traceback:

```bash
dyana run path/to/sample.wav --out-dir out --debug
```

## CLI surface

The top-level entrypoint is:

```bash
dyana <command> [options]
```

Current subcommands include:

- `run` for the end-to-end pipeline
- `asr-setup` for ASR-related setup
- `decode` for decoding-focused workflows
- `evidence` for evidence extraction steps
- `iterate` for iteration loop workflows
- `eval` for evaluation harnesses and scorecards
- `tune` for tuning-related experiments

## Development

Source code lives under `src/dyana`, tests live under `tests`, and the Sphinx documentation source lives under `docs`.

Run the test suite with:

```bash
pytest
```

If you only want a narrower smoke path:

```bash
pytest tests/cli/commands/test_run.py tests/test_cli_main.py tests/errors tests/pipeline
```

## Project direction

DYANA is being developed around a simple idea: audio annotation pipelines should be explicit, reproducible, and easy to inspect. The goal is not only to generate outputs, but to make it clear which steps ran, which failed, which artifacts were produced, and how later stages depended on earlier ones.
