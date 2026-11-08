import os, time
import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm

DB = dict(host="localhost", port=5432, user="postgres",
          password="postgres", dbname="busqueda")
K = 10

FILTROS = {
    "sin_filtro":   ("TRUE",                    1.0000),
    "bucket_500":   ("bucket < 500",            0.4989),
    "bucket_100":   ("bucket < 100",            0.0993),
    "bucket_10":    ("bucket < 10",             0.01007),
    "bucket_1":     ("bucket < 1",              0.00100),
    "cat_World":    ("categoria = 'World'",     0.2500),
    "cat_Sports":   ("categoria = 'Sports'",    0.2500),
    "cat_Business": ("categoria = 'Business'",  0.2500),
    "cat_SciTech":  ("categoria = 'Sci/Tech'",  0.2500),
}

# margen = cuántos candidatos pide al índice; ef_search = tamaño de la lista
CONFIGS = [
    {"nombre": "post_ingenuo",     "margen": 10,   "ef_search": 40},
    {"nombre": "post_margen_100",  "margen": 100,  "ef_search": 100},
    {"nombre": "post_margen_1000", "margen": 1000, "ef_search": 1000},
]

queries = np.load("data/queries.npy")
gt = pd.read_csv("results/ground_truth.csv")

conn = psycopg2.connect(**DB)
cur = conn.cursor()

registros = []
for fname, (cond, sel) in FILTROS.items():
    gt_f = gt[gt["filtro"] == fname]
    for cfg in CONFIGS:
        cur.execute(f"SET hnsw.ef_search = {cfg['ef_search']};")
        tiempos, recalls, filas = [], [], []
        for qid, q in enumerate(tqdm(queries, desc=f"{fname}/{cfg['nombre']}", leave=False)):
            q_str = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
            # Post-filter CORRECTO: se piden {margen} candidatos, se filtra,
            # y se ordenan los que sobreviven por distancia.
            sql = f"""
                WITH cand AS (
                    SELECT id, embedding <=> %s::vector AS dist
                    FROM contracts
                    ORDER BY embedding <=> %s::vector
                    LIMIT {cfg['margen']}
                )
                SELECT c.id
                FROM cand c JOIN contracts t ON t.id = c.id
                WHERE {cond}
                ORDER BY c.dist
                LIMIT {K};
            """
            t0 = time.perf_counter()
            cur.execute(sql, (q_str, q_str))
            tiempos.append((time.perf_counter() - t0) * 1000)
            ids = {row[0] for row in cur.fetchall()}
            filas.append(len(ids))
            gt_ids = set(gt_f[gt_f["query_id"] == qid]["doc_id"].head(K).tolist())
            recalls.append(len(ids & gt_ids) / len(gt_ids) if gt_ids else 0.0)

        registros.append({
            "filtro": fname,
            "selectividad": sel,
            "config": cfg["nombre"],
            "filas_devueltas_media": round(float(np.mean(filas)), 2),
            "recall@10": round(float(np.mean(recalls)), 4),
            "latencia_media_ms": round(float(np.mean(tiempos)), 3),
        })

cur.close(); conn.close()
df = pd.DataFrame(registros)
print(df.to_string(index=False))
df.to_csv("results/expA_postfilter.csv", index=False)
print("\nGuardado en results/expA_postfilter.csv")