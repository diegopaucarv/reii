"""Herramientas de diagnóstico del pipeline REII.

Los comandos se ejecutan desde el CLI:

    python -m reii.diag.cli <comando> [flags]     # entorno instalado / contenedor
    python diag.py <comando> [flags]              # host sin `pip install -e .`

Contrato de cada módulo de comando (``trials``, ``optuna_db``, ``uce_size``,
``preflight``, ``db_hygiene``, ``multiverse``, ``selection_audit``):

  * ``run(argv: list[str] | None = None) -> int`` — parsea SUS flags y devuelve
    el código de salida. Es lo único que necesita ``cli.py``.
  * ``build_parser() -> argparse.ArgumentParser`` — el parser completo del
    comando, para que ``<cmd> --help`` lo imprima.

Nada de este paquete se importa desde ``main_workflow``: el pipeline no paga su
coste (y los comandos no arrastran Stanza salvo que su lógica lo exija).
"""
