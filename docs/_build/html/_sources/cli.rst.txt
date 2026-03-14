Command Line Interface
======================

Entrypoint
----------

The package exposes a single console entrypoint:

.. code-block:: bash

   dyana <command> [options]

Top-level subcommands are registered in ``dyana.cli.main`` and currently include:

- ``run`` for the end-to-end pipeline
- ``decode`` for decoding-focused workflows
- ``evidence`` for evidence extraction steps
- ``iterate`` for iteration loop workflows
- ``eval`` for evaluation harnesses and scorecards
- ``tune`` for tuning-related tasks

How the CLI is structured
-------------------------

The command modules live under ``dyana.cli.commands``. Each subcommand module contributes
its parser setup with ``add_subparser(...)`` and a matching ``run(...)`` implementation.

Reference
---------

See the generated API page for the CLI entrypoint:

- :doc:`api/generated/dyana.cli.main`
