import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

# ---------------- Experimento 0 ----------------
df0 = pd.read_csv("results/exp0_baseline.csv")

# Latencia media vs K, por configuración
piv = df0.pivot(index="K", columns="config", values="latencia_media_ms")
piv.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Latencia media (ms)")
plt.xlabel("K")
plt.title("Experimento 0 — Latencia exacta vs HNSW")
plt.yscale("log")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/fig_exp0_latencia.png")
plt.close()

# Recall@K vs K
piv = df0.pivot(index="K", columns="config", values="recall@K")
piv.plot(kind="line", marker="o", figsize=(8, 5))
plt.ylabel("Recall@K")
plt.xlabel("K")
plt.ylim(0.95, 1.01)
plt.title("Experimento 0 — Recall exacta vs HNSW")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/fig_exp0_recall.png")
plt.close()

# ---------------- Experimento A ----------------
dfA = pd.read_csv("results/expA_postfilter.csv")

# Recall vs selectividad (el gráfico clave del trabajo)
fig, ax = plt.subplots(figsize=(8, 5))
for cfg, grupo in dfA.groupby("config"):
    grupo = grupo.sort_values("selectividad")
    ax.plot(grupo["selectividad"], grupo["recall@10"],
            marker="o", label=cfg)
ax.set_xscale("log")
ax.set_xlabel("Selectividad del filtro (fracción)")
ax.set_ylabel("Recall@10")
ax.set_title("Experimento A — Recall vs selectividad")
ax.grid(alpha=0.3, which="both")
ax.legend()
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig("results/figures/fig_expA_recall_selectividad.png")
plt.close()

# Filas devueltas vs selectividad
fig, ax = plt.subplots(figsize=(8, 5))
for cfg, grupo in dfA.groupby("config"):
    grupo = grupo.sort_values("selectividad")
    ax.plot(grupo["selectividad"], grupo["filas_devueltas_media"],
            marker="s", label=cfg)
ax.set_xscale("log")
ax.set_xlabel("Selectividad del filtro (fracción)")
ax.set_ylabel("Filas devueltas (media)")
ax.set_title("Experimento A — Filas devueltas vs selectividad")
ax.axhline(10, color="gray", linestyle="--", linewidth=0.8,
           label="K = 10 esperadas")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/fig_expA_filas_selectividad.png")
plt.close()

# Latencia vs selectividad
fig, ax = plt.subplots(figsize=(8, 5))
for cfg, grupo in dfA.groupby("config"):
    grupo = grupo.sort_values("selectividad")
    ax.plot(grupo["selectividad"], grupo["latencia_media_ms"],
            marker="^", label=cfg)
ax.set_xscale("log")
ax.set_xlabel("Selectividad del filtro (fracción)")
ax.set_ylabel("Latencia media (ms)")
ax.set_title("Experimento A — Latencia vs selectividad")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/fig_expA_latencia_selectividad.png")
plt.close()

print("Gráficas generadas en results/figures/")