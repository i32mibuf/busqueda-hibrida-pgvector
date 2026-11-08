import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

SEED = 42
N_QUERIES = 200
MODELO = "sentence-transformers/all-MiniLM-L6-v2"
NOMBRES_CAT = ["World", "Sports", "Business", "Sci/Tech"]

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(SEED)

# 1. Datos
ds = load_dataset("fancyzhx/ag_news")
train, test = ds["train"], ds["test"]
textos = train["text"]
print(f"Documentos: {len(textos)}")

# 2. Modelo
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Dispositivo: {device}")
model = SentenceTransformer(MODELO, device=device)

# 3. Embeddings de los documentos
emb = model.encode(
    textos, batch_size=256, normalize_embeddings=True,
    show_progress_bar=True, convert_to_numpy=True,
).astype(np.float32)

# 4. Embeddings de las consultas
idx_q = rng.choice(len(test), size=N_QUERIES, replace=False)
textos_q = [test["text"][i] for i in idx_q]
emb_q = model.encode(
    textos_q, batch_size=256, normalize_embeddings=True,
    convert_to_numpy=True,
).astype(np.float32)

# 5. Metadatos
df = pd.DataFrame({
    "id": np.arange(len(textos)),
    "contenido": textos,
    "categoria": [NOMBRES_CAT[l] for l in train["label"]],
    "bucket": rng.integers(0, 1000, size=len(textos)),
})

# 6. Guardar
np.save("data/embeddings.npy", emb)
np.save("data/queries.npy", emb_q)
df.to_csv("data/documentos.csv", index=False)
pd.DataFrame({"query_id": np.arange(N_QUERIES), "texto": textos_q}) \
  .to_csv("data/queries_texto.csv", index=False)

# 7. Comprobaciones
print("\n--- Verificación ---")
print("Forma embeddings:", emb.shape)
print("Norma media (debe ser ~1):", np.linalg.norm(emb, axis=1).mean())
print("Distribución por categoría:\n", df["categoria"].value_counts(normalize=True))
for umbral in (500, 100, 10, 1):
    print(f"bucket < {umbral:>3}: {(df['bucket'] < umbral).mean():.4%}")