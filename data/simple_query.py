import sqlite3

conn = sqlite3.connect("C:/Users/Julia/Downloads/reii/data/workflow_data.db")
c = conn.cursor()

# 1. uces - schema + sample
c.execute("PRAGMA table_info(uces)")
print("=== uces schema ===")
cols = c.fetchall()
print("Columns:", [col[1] for col in cols])
c.execute("SELECT COUNT(*) FROM uces WHERE embedding IS NULL")
print("Rows with NULL embedding:", c.fetchone()[0])

# 2. pairwise_stability
c.execute("PRAGMA table_info(pairwise_stability)")
print("\n=== pairwise_stability schema ===")
ps_cols = c.fetchall()
print("Columns:", [col[1] for col in ps_cols])

# 3. shap_analysis
c.execute("PRAGMA table_info(shap_analysis)")
print("\n=== shap_analysis schema ===")
sa_cols = c.fetchall()
print("Columns:", [col[1] for col in sa_cols])

# 4. proyeccion
c.execute("PRAGMA table_info(proyeccion)")
print("\n=== proyeccion schema ===")
pj_cols = c.fetchall()
print("Columns:", [col[1] for col in pj_cols])

conn.close()
