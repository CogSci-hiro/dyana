.. title:: DYANA

.. raw:: html

   <div class="dyana-title-logo-wrap">
     <img class="dyana-title-logo" src="_static/logo-tight.svg" alt="DYANA logo" />
   </div>

.. raw:: html

   <div class="dyana-hero">
     <p class="dyana-eyebrow">DYadic Annotation of Naturalistic Audio</p>
     <h1>Reliable conversational audio annotation with transparent evidence and constrained decoding.</h1>
     <p class="dyana-lead">
       DYANA combines evidence extraction, structured state decoding, evaluation,
       and iteration loops into a lightweight Python toolkit for dyadic audio workflows.
     </p>
   </div>

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Get from install to a first pipeline run and understand the generated artifacts.

   .. grid-item-card:: CLI
      :link: cli
      :link-type: doc

      See the `dyana` command layout, main subcommands, and how the pieces connect.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Browse the package modules, classes, and functions generated from the codebase.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Evidence-first pipeline

      The pipeline extracts aligned evidence tracks, fuses them into framewise scores,
      decodes constrained state paths, and writes reusable artifacts.

   .. grid-item-card:: Evaluation and tuning

      DYANA includes synthetic harnesses, scorecards, and tuning helpers so decoder
      changes can be evaluated instead of guessed.

.. toctree::
   :hidden:
   :maxdepth: 2

   quickstart
   cli
   architecture
   project_notes
   api/index
