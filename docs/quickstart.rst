Quickstart
==========

Installation
------------

Install the package in editable mode with the documentation extras when you want to build this site locally:

.. code-block:: bash

   python -m pip install -e ".[docs,dev]"

Core workflow
-------------

DYANA is organized around a small number of pipeline stages:

1. Extract evidence tracks from audio.
2. Fuse evidence into decoder scores.
3. Decode a constrained sequence of conversational states.
4. Export artifacts such as NumPy arrays, JSON diagnostics, and Praat TextGrid files.

Run the pipeline
----------------

Use the top-level CLI to run the default end-to-end workflow on an input audio file:

.. code-block:: bash

   dyana run INPUT.wav --out-dir artifacts/run_001

When you want higher recall before ASR, enable the recall-first profile and optional pyannote proposals:

.. code-block:: bash

   dyana run INPUT.wav --out-dir artifacts/run_001 --profile recall-first --pyannote --vad-backend all

Pyannote is optional. It is only imported when explicitly enabled, and token lookup follows this order:

- ``--pyannote-token``
- ``HF_TOKEN``
- ``HUGGINGFACE_TOKEN``

The pipeline writes:

- evidence tracks under ``artifacts/run_001/evidence/``
- decoded states and IPU summaries under ``artifacts/run_001/decode/``
- a ``.TextGrid`` export at the run root
- diagnostics JSON alongside the decode outputs

Phase 1 notes
-------------

- WebRTC VAD is now baseline / optional evidence rather than a required core cue.
- Pyannote proposal tracks are coarse speech and anonymous-speaker hints on DYANA's 10 ms grid.
- DYANA still owns stereo reasoning, final A/B mapping, and robust ``OVL`` / ``LEAK`` handling.

Explore next
------------

- See :doc:`cli` for the command surface.
- See :doc:`architecture` for how the package modules fit together.
- See :doc:`api/index` for module-level reference material.
