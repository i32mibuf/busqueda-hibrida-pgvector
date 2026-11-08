import os
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

DB = dict(host="localhost", port=5432, user="postgres",
          password="postgres", dbname="busqueda")

print("Cargando datos...")
emb = np.load("data/embeddings.npy")
df  = pd.read_csv("data/documentos.csv")

assert len(df) == emb.shape[0], "Desajuste entre CSV y embeddings"
print(f"Documentos: {len(df)}  Dimensiones: {emb.shape[1]}")

conn = psycopg2.connect(**DB)
cur  = conn.cursor()

cur.execute("DROP TABLE IF EXISTS contracts CASCADE;")
cur.execute("""
    CREATE TABLE contracts (
        id          INTEGER PRIMARY KEY,
        contenido   TEXT NOT NULL,
        categoria   TEXT NOT NULL,
        bucket      INTEGER NOT NULL,
        embedding   vector(384) NOT NULL
    );
""")

rows = [
    (int(r.id), r.contenido, r.categoria, int(r.bucket), emb[i].tolist())
    for i, r in df.iterrows()
]

print("Insertando en lotes de 1000...")
for i in tqdm(range(0, len(rows), 1000)):
    execute_values(
        cur,
        "INSERT INTO contracts (id, contenido, categoria, bucket, embedding) VALUES %s",
        rows[i:i+1000],
        page_size=1000,
    )
conn.commit()

# Índices B-tree auxiliares (para el Experimento B)
print("Creando índices B-tree auxiliares...")
cur.execute("CREATE INDEX contracts_bucket_idx    ON contracts (bucket);")
cur.execute("CREATE INDEX contracts_categoria_idx ON contracts (categoria);")
conn.commit()

# Verificación
cur.execute("SELECT COUNT(*) FROM contracts;")
print("Filas en la tabla:", cur.fetchone()[0])
cur.execute("SELECT pg_size_pretty(pg_total_relation_size('contracts'));")
print("Tamaño total de la tabla:", cur.fetchone()[0])

cur.close(); conn.close()
print("OK")