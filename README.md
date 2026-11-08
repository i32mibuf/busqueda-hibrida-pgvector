# Búsqueda híbrida en PostgreSQL con pgvector

Estudio de cómo se comporta la búsqueda por similitud vectorial cuando se combina con un filtro SQL (pre-filtering y post-filtering) en PostgreSQL con pgvector. Se usan 120.000 noticias de AG News con embeddings de 384 dimensiones y se mide recall@10, filas devueltas y latencia con distintas selectividades del filtro.

## Estructura

- `docker-compose.yml` y `init.sql`: PostgreSQL 17 con pgvector 0.8.7
- `requirements.txt`: dependencias de Python
- `scripts/`: generación de embeddings, carga, índice HNSW, experimentos y gráficas
- `results/`: resultados de los experimentos en CSV

## Cómo reproducirlo

```bash
docker compose up -d
pip install -r requirements.txt

python scripts/01_generar_embeddings.py
python scripts/02_cargar_datos.py
python scripts/03_ground_truth.py
python scripts/04_crear_indice_hnsw.py
python scripts/05_exp0_baseline.py
python scripts/06_expA_postfilter.py
python scripts/07_plots.py
```

Los embeddings no se incluyen en el repositorio por su tamaño; se regeneran con el primer script (unos 2 minutos).

## Entorno

PostgreSQL 17.11, pgvector 0.8.7, Python 3.13, Docker Compose.