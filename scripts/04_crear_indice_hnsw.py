import time
import psycopg2

DB = dict(host="localhost", port=5432, user="postgres",
          password="postgres", dbname="busqueda")
conn = psycopg2.connect(**DB); conn.autocommit = True
cur = conn.cursor()

print("Creando índice HNSW (m=16, ef_construction=64)...")
t0 = time.perf_counter()
cur.execute("""
    CREATE INDEX contracts_embedding_hnsw
    ON contracts USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")
print(f"Tiempo de construcción: {time.perf_counter()-t0:.1f} s")

cur.execute("SELECT pg_size_pretty(pg_relation_size('contracts_embedding_hnsw'));")
print("Tamaño del índice HNSW:", cur.fetchone()[0])

# Verificar que el planificador lo usa
cur.execute("LOAD 'vector';")
cur.execute("SHOW hnsw.ef_search;")
print("ef_search:", cur.fetchone()[0])
cur.execute("SHOW hnsw.iterative_scan;")
print("iterative_scan:", cur.fetchone()[0])

cur.execute("""
    EXPLAIN (ANALYZE, BUFFERS)
    SELECT id FROM contracts
    ORDER BY embedding <=> (SELECT embedding FROM contracts WHERE id = 0)
    LIMIT 10;
""")
for row in cur.fetchall():
    print(row[0])

cur.close(); conn.close()