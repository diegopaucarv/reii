import pathlib
p = pathlib.Path("src/reii/main_workflow.py")
if not p.exists():
    p = pathlib.Path("main_workflow.py")
src = p.read_text(encoding="utf-8")
print(f"Archivo: {p}  ({len(src):,} chars)\n")
checks = {
    "Config.pseudocount = 0.01":       "pseudocount: float = 0.01" in src,
    "Config.pseudocount = 0.0":        "pseudocount: float = 0.0" in src,
    "__main__ pseudocount=0.1":        "pseudocount=0.1" in src,
    "__main__ pseudocount=0.0":        "pseudocount=0.0" in src,
    "AFC: + self.config.pseudocount":  "+ self.config.pseudocount" in src,
    "AFC: randomized_svd":             "randomized_svd" in src,
    "AFC: scipy_svd":                  "from scipy.linalg import svd" in src,
    "_primer_factor: + pc * m":        "+ pc * m" in src,
    "dok_matrix":                      "dok_matrix" in src,
    "coo_matrix":                      "coo_matrix" in src,
    "min_term_abs_freq":               "min_term_abs_freq" in src,
    "uce_target_size":                 "uce_target_size" in src,
    "class Database":                  "class Database" in src,
    "PatternSearch":                   "PatternSearch" in src,
    "import optuna":                   "import optuna" in src,
    "_project_liminal_uces":           "_project_liminal_uces" in src,
    "use_liminal_projection":          "use_liminal_projection" in src,
}
for name, present in checks.items():
    print(f"{'OK ' if present else 'NO '}  {name}")
