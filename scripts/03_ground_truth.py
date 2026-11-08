import os
import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm

DB = dict(host="localhost", port=5432, user="postgres",
          password="postgres", dbname="busqueda")
K = 10

FILTROS = {
    "sin_filtro":  "TRUE",
    "bucket_500":  "bucket < 500",
    "bucket_100":  "bucket < 100",
    "bucket_10":   "bucket < 10",
    "bucket_1":    "bucket < 1",
    "cat_World":   "categoria = 'World'",
    "cat_Sports":  "categoria = 'Sports'",
    "cat_Business":"categoria = 'Business'",
    "cat_SciTech": "categoria = 'Sci/Tech'",
}

queries = np.load("data/queries.npy")
print(f"Consultas: {len(queries)}  Filtros: {len(FILTROS)}  K={K}")

conn = psycopg2.connect(**DB)
conn.autocommit = False
cur = conn.cursor()
# Forzar búsqueda exacta en toda la sesión
cur.execute("SET enable_indexscan = off;")
cur.execute("SET enable_bitmapscan = off;")

registros = []
for qid, q in enumerate(tqdm(queries, desc="Consultas")):
    q_str = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
    for nombre, cond in FILTROS.items():
        sql = f"""
            SELECT id, embedding <=> %s::vector AS dist
            FROM contracts
            WHERE {cond}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        cur.execute(sql, (q_str, q_str, K))
        for rank, (doc_id, dist) in enumerate(cur.fetchall(), start=1):
            registros.append({
                "query_id": qid, "filtro": nombre,
                "rank": rank, "doc_id": doc_id, "distancia": float(dist),
            })

conn.rollback()  # no queremos cambios, solo lecturas
cur.close(); conn.close()

os.makedirs("results", exist_ok=True)
pd.DataFrame(registros).to_csv("results/ground_truth.csv", index=False)
print(f"Ground truth guardada: {len(registros)} filas")
print("Distribución por filtro:")
print(pd.DataFrame(registros).groupby("filtro").size())