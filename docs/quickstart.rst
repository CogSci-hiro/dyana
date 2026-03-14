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

The pipeline writes:

- evidence tracks under ``artifacts/run_001/evidence/``
- decoded states and IPU summaries under ``artifacts/run_001/decode/``
- a ``.TextGrid`` export at the run root
- diagnostics JSON alongside the decode outputs

Explore next
------------

- See :doc:`cli` for the command surface.
- See :doc:`architecture` for how the package modules fit together.
- See :doc:`api/index` for module-level reference material.
