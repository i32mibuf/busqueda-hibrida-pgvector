import os, time
import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm

DB = dict(host="localhost", port=5432, user="postgres",
          password="postgres", dbname="busqueda")

queries = np.load("data/queries.npy")
gt = pd.read_csv("results/ground_truth.csv")
gt_sin = gt[gt["filtro"] == "sin_filtro"]

KS = [10, 50, 100]
configs = [
    {"nombre": "exacto",     "enable_indexscan": "off", "ef_search": None},
    {"nombre": "hnsw_ef40",  "enable_indexscan": "on",  "ef_search": 40},
    {"nombre": "hnsw_ef100", "enable_indexscan": "on",  "ef_search": 100},
]

conn = psycopg2.connect(**DB)
cur = conn.cursor()

registros = []
for K in KS:
    for cfg in configs:
        cur.execute(f"SET enable_indexscan = {cfg['enable_indexscan']};")
        if cfg["ef_search"] is not None:
            cur.execute(f"SET hnsw.ef_search = {cfg['ef_search']};")

        tiempos, recalls = [], []
        for qid, q in enumerate(tqdm(queries, desc=f"K={K} {cfg['nombre']}", leave=False)):
            q_str = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
            t0 = time.perf_counter()
            cur.execute(
                "SELECT id FROM contracts ORDER BY embedding <=> %s::vector LIMIT %s;",
                (q_str, K),
            )
            tiempos.append((time.perf_counter() - t0) * 1000)
            ids = {row[0] for row in cur.fetchall()}
            gt_ids = set(gt_sin[gt_sin["query_id"] == qid]["doc_id"].head(K).tolist())
            recalls.append(len(ids & gt_ids) / len(gt_ids) if gt_ids else 0.0)

        registros.append({
            "K": K,
            "config": cfg["nombre"],
            "latencia_media_ms": round(float(np.mean(tiempos)), 3),
            "latencia_p50_ms":   round(float(np.median(tiempos)), 3),
            "latencia_p95_ms":   round(float(np.percentile(tiempos, 95)), 3),
            "recall@K":          round(float(np.mean(recalls)), 4),
        })

cur.close(); conn.close()
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(registros)
print(df.to_string(index=False))
df.to_csv("results/exp0_baseline.csv", index=False)
print("\nGuardado en results/exp0_baseline.csv")