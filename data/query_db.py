import sqlite3

db = "C:/Users/Julia/Downloads/reii/data/workflow_data.db"
conn = sqlite3.connect(db)
c = conn.cursor()

# First, find all tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print("=== TABLES ===")
for t in tables:
    print(f"  {t}")

# ---- 1. uces ----
print("\n\n=== uces ===")
c.execute("PRAGMA table_info(uces)")
cols = c.fetchall()
print(f"\nColumns ({len(cols)}):")
for col in cols:
    print(f"  {col}")

emb_found = any("embedding" in col[1] for col in cols)
print(f"\nEmbedding column: {'FOUND' if emb_found else 'NOT FOUND'}")

if emb_found:
    emb_col = [col[1] for col in cols if "embedding" in col[1]][0]
    c.execute(f"SELECT COUNT(*) FROM uces WHERE {emb_col} IS NULL")
    nulls = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM uces")
    total = c.fetchone()[0]
    print(f"  NULL embeddings: {nulls}/{total}")

print("\nRow 0 sample (first 15 cols):")
c.execute("SELECT * FROM uces LIMIT 1")
row = c.fetchone()
for i in range(min(15, len(cols))):
    val = row[i]
    v = "NULL" if val is None else str(val)[:40]
    print(f"    {cols[i][1]}: {v}")

# ---- 2. pairwise_stability (if exists) ----
print("\n\n=== pairwise_stability ===")
if "pairwise_stability" in tables:
    c.execute("PRAGMA table_info(pairwise_stability)")
    cols = c.fetchall()
    print(f"Columns ({len(cols)}):")
    for col in cols:
        print(f"  {col}")
    c.execute("SELECT * FROM pairwise_stability LIMIT 2")
    rows = c.fetchall()
    col_names = [col[1] for col in cols]
    for i, row in enumerate(rows):
        print(f"\n  Row {i}:")
        for name, val in zip(col_names, row):
            v = "NULL" if val is None else str(val)[:30]
            print(f"    {name}: {v}")
else:
    print("  Table DOES NOT EXIST")

# ---- 3. shap_analysis (if exists) ----
print("\n\n=== shap_analysis ===")
if "shap_analysis" in tables:
    c.execute("PRAGMA table_info(shap_analysis)")
    cols = c.fetchall()
    col_names = [col[1] for col in cols]
    print(f"Columns ({len(cols)}):")
    for col in cols:
        print(f"  {col}")

    print(f"\nColumn names: {col_names}")

    # Check target columns
    for t in ["y_true", "oof_predictions", "confusion_matrix"]:
        if t in col_names:
            c.execute(f"SELECT COUNT(*) FROM shap_analysis WHERE {t} IS NULL")
            nulls = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM shap_analysis")
            total = c.fetchone()[0]
            print(f"  {t}: EXISTS — NULLs: {nulls}/{total}")
        else:
            print(f"  {t}: MISSING")

    c.execute("SELECT COUNT(*) FROM shap_analysis")
    print(f"Total runs: {c.fetchone()[0]}")

else:
    print("  Table DOES NOT EXIST")

# ---- 4. proyeccion (if exists) ----
print("\n\n=== proyeccion ===")
if "proyeccion" in tables:
    c.execute("PRAGMA table_info(proyeccion)")
    cols = c.fetchall()
    col_names = [col[1] for col in cols]
    print(f"Columns ({len(cols)}):")
    for col in cols:
        print(f"  {col}")

    print(f"\nColumn names: {col_names}")

    for t in ["voc", "col_coords", "col_std", "kept_cols_idx"]:
        if t in col_names:
            c.execute(f"SELECT COUNT(*) FROM proyeccion WHERE {t} IS NULL")
            nulls = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM proyeccion")
            total = c.fetchone()[0]
            print(f"  {t}: EXISTS — NULLs: {nulls}/{total}")
        else:
            print(f"  {t}: MISSING")

    c.execute("SELECT COUNT(*) FROM proyeccion")
    print(f"Total runs: {c.fetchone()[0]}")
else:
    print("  Table DOES NOT EXIST")

print("\n\nDone!")
conn.close()
