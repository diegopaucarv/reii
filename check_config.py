import pathlib, re
p = pathlib.Path("src/reii/main_workflow.py")
src = p.read_text(encoding="utf-8")

m = re.search(r"alceste_config\s*=\s*Config\((.*?)\n\s*\)", src, re.S)
if not m:
    print("NO se encontro alceste_config = Config(...)")
else:
    print("=== Config activo ===")
    for line in m.group(1).splitlines():
        line = line.strip().rstrip(",")
        if any(k in line for k in [
            "optimize", "classification_mode",
            "use_liminal_projection", "liminal_projection_min_ratio",
            "pseudocount", "uce_target_size", "min_uce_words",
            "min_forms_uc", "tsj", "min_term_abs_freq",
        ]):
            print(f"  {line}")

print()
print("=== Duplicados de optimize* en __main__ ===")
count = 0
for i, line in enumerate(src.splitlines(), 1):
    if re.search(r"^\s*optimize(_trials|_sampler|_pruner|_storage|_study_name|_preference|_coverage_gate|_directions|_multivariate|_n_startup_trials|_prune_n_startup)?\s*=", line):
        count += 1
        print(f"  {i:>5}: {line.rstrip()}")
print(f"  total: {count} lineas")
