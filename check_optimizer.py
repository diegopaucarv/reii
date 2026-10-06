import pathlib, re
p = pathlib.Path("src/reii/main_workflow.py")
src = p.read_text(encoding="utf-8")

m = re.search(r"^class Optimizador\b.*?(?=^class |\Z)", src, re.M | re.S)
if not m:
    print("NO se encontro class Optimizador")
else:
    body = m.group(0)
    print(f"Optimizador: {len(body)} chars")
    print()
    for pat in ["_PatternSearch(", "optuna.create_study",
                "study.optimize", "_evaluate_trial",
                "def objetivo", "def optimizar"]:
        present = pat in body
        print(f"  {'OK ' if present else 'NO '}  {pat}")

print()
print("Lineas con 'PatternSearch':")
for i, line in enumerate(src.splitlines(), 1):
    if "PatternSearch" in line:
        print(f"  {i:>5}: {line.rstrip()[:120]}")
