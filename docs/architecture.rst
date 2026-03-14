Architecture
============

Package layout
--------------

The codebase is organized into a few coherent layers:

- ``dyana.core`` provides shared data types, timebase handling, caching, and resampling utilities.
- ``dyana.evidence`` computes aligned evidence tracks such as VAD, prosody, energy, leakage, and synthetic helpers.
- ``dyana.decode`` contains the constrained decoder, state space definitions, transition logic, and IPU extraction.
- ``dyana.pipeline`` assembles the end-to-end workflow from audio input to exported artifacts.
- ``dyana.eval`` and ``dyana.iterate`` support scorecards, tuning, uncertainty analysis, and iterative improvement loops.
- ``dyana.io`` handles serialization and external file formats such as TextGrid.
- ``dyana.errors`` centralizes error types, guards, and reporting utilities.

Processing flow
---------------

.. code-block:: text

   audio
     -> evidence tracks
     -> fused decoder scores
     -> constrained state path
     -> IPU segments + diagnostics
     -> persisted artifacts

Representative entrypoint
-------------------------

``dyana.pipeline.run_pipeline.run_pipeline`` is the clearest single reference point for the
full workflow. It computes evidence tracks, builds an ``EvidenceBundle``, decodes states,
extracts IPUs, writes artifact files, and returns a summary dictionary.

API cross-links
---------------

- :mod:`dyana.pipeline.run_pipeline`
- :mod:`dyana.evidence.base`
- :mod:`dyana.decode.decoder`
