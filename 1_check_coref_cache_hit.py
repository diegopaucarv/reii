#!/usr/bin/env python3
"""
Script 1: Verifica que el fix de caché de coref funciona.
Lee los logs de la ejecución y reporta hits/misses.

USO (después de ejecutar main_workflow.py dos veces):
  python 1_check_coref_cache_hit.py --log1 output.log --log2 output_2.log

O simplemente busca en los logs existentes:
  python 1_check_coref_cache_hit.py --reii-dir /ruta/a/reii

ESPERA VER:
  - Primera corrida: Y extraídos con Stanza = <número de docs>
  - Segunda corrida: Y extraídos = 0 (todos desde caché)
"""

import re
import sys
from argparse import ArgumentParser
from pathlib import Path


def extract_coref_hits_misses(log_content):
    """Busca el resumen [COREF context=N] en el log."""
    # Patrón: "[COREF context=2] cadenas: 15 docs desde caché, 3 extraídos con Stanza"
    match = re.search(
        r"\[COREF context=(\d+)\]\s+cadenas:\s+(\d+)\s+docs desde\s+caché,\s+(\d+)\s+extraídos",
        log_content,
    )
    if match:
        ctx, hits, misses = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        return hits, misses
    return None, None


def find_latest_logs(reii_dir):
    """Busca output.log en el directorio REII."""
    reii_dir = Path(reii_dir)
    log_path = reii_dir / "output.log"

    if log_path.exists():
        return str(log_path), None
    else:
        print(f"⚠ No se encontró {log_path}")
        return None, None


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reii-dir", help="Ruta al directorio de REII (para buscar logs)"
    )
    parser.add_argument("--log1", help="Primer log (corrida sin caché)")
    parser.add_argument("--log2", help="Segundo log (corrida con caché)")
    args = parser.parse_args()

    print("=" * 70)
    print("SCRIPT 1: Verificar fix de caché de coref")
    print("=" * 70)

    # Determinar paths de logs
    if args.log1 and args.log2:
        log1_path = Path(args.log1)
        log2_path = Path(args.log2)
    elif args.reii_dir:
        log1_path, _ = find_latest_logs(args.reii_dir)
        log2_path = None
        if not log1_path:
            print("\n✗ No se encontraron logs en", args.reii_dir)
            sys.exit(1)
        print(f"\n→ Usando log: {log1_path}")
        print(
            "  (Para verificar con caché, ejecuta main_workflow.py de nuevo y sube el nuevo log)"
        )
    else:
        parser.print_help()
        sys.exit(1)

    # Leer primer log
    if not Path(log1_path).exists():
        print(f"\n✗ Log no existe: {log1_path}")
        sys.exit(1)

    log1_content = Path(log1_path).read_text(encoding="utf-8", errors="ignore")
    h1, m1 = extract_coref_hits_misses(log1_content)

    if h1 is None:
        print(f"\n⚠ No se encontró el resumen de caché en {log1_path}")
        print("  Busco en el log...")
        if "[COREF context=" in log1_content:
            print("  ✓ Encontré [COREF context= en el log, pero el patrón no coincide")
            lines = [l for l in log1_content.split("\n") if "[COREF" in l]
            for l in lines[:3]:
                print(f"    {l[:100]}")
        else:
            print("  ✗ No hay líneas [COREF context= en el log")
            print("  ¿Se aplicaron los patches correctamente?")
        sys.exit(1)

    print(f"\nCORRIDA 1 ({log1_path}):")
    print(f"  {h1} docs desde caché, {m1} extraídos con Stanza")

    # Si hay log 2, analizarlo
    if log2_path and Path(log2_path).exists():
        log2_content = Path(log2_path).read_text(encoding="utf-8", errors="ignore")
        h2, m2 = extract_coref_hits_misses(log2_content)

        if h2 is not None:
            print(f"\nCORRIDA 2 ({log2_path}):")
            print(f"  {h2} docs desde caché, {m2} extraídos con Stanza")

            print("\n" + "=" * 70)
            print("DIAGNÓSTICO:")

            if m2 == 0:
                print("✓ FIX FUNCIONANDO PERFECTAMENTE")
                print(f"  Corrida 1: {m1} docs extraídos")
                print(f"  Corrida 2: {h2} desde caché, {m2} nuevos")
                if m1 > 0:
                    print(f"  Speedup: 100% (sin re-extracciones)")
            elif m2 < m1 * 0.1:
                print(f"~ FIX PARCIALMENTE FUNCIONAL")
                print(f"  {m2} / {m1} = {100 * m2 / m1:.0f}% se re-extraen")
                print(f"  Posibles causas: corpus cambió, uce_target_size cambió")
            else:
                print(f"✗ FIX NO FUNCIONA")
                print(f"  {m2} / {m1} = {100 * m2 / m1:.0f}% aún se re-extraen")
                print(f"  Comprueba:")
                print(f"    - ¿Se aplicó el patch reii_fixes.patch?")
                print(f"    - ¿El caché existe en output/coref_chains.json?")
                print(f"    - ¿El corpus es exactamente el mismo?")
        else:
            print(f"⚠ No se pudo extraer datos de corrida 2")
    else:
        print("\n" + "=" * 70)
        print("SIGUIENTE PASO:")
        print("  1. Ejecuta main_workflow.py una segunda vez (sin cambios)")
        print("  2. Guarda el nuevo log como 'output_2.log'")
        print("  3. Corre de nuevo:")
        print(
            "     python 1_check_coref_cache_hit.py --log1 output.log --log2 output_2.log"
        )


if __name__ == "__main__":
    main()
