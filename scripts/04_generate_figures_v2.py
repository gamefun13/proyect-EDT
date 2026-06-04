import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =========================
# 1. RUTAS V2
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

METRICS_DIR = os.path.join(BASE_DIR, "outputs_v2", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs_v2", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_COMPARISON_PATH = os.path.join(METRICS_DIR, "model_comparison.csv")
AREA_BY_CLASS_PATH = os.path.join(METRICS_DIR, "area_by_class_v2.csv")
BEST_CM_PATH = os.path.join(METRICS_DIR, "confusion_matrix_KNN.csv")


# =========================
# 2. CARGAR DATOS
# =========================

model_df = pd.read_csv(MODEL_COMPARISON_PATH)
area_df = pd.read_csv(AREA_BY_CLASS_PATH)
cm_df = pd.read_csv(BEST_CM_PATH, index_col=0)

print("Model comparison v2:")
print(model_df)

print("\nArea by class v2:")
print(area_df)

print("\nConfusion matrix KNN v2:")
print(cm_df)


# =========================
# 3. GRÁFICO COMPARATIVO DE MODELOS
# =========================

metrics = ["accuracy", "kappa", "precision_macro", "recall_macro", "f1_macro"]

model_plot_df = model_df[["model"] + metrics].copy()

x = np.arange(len(model_plot_df["model"]))
width = 0.15

plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):
    plt.bar(x + i * width, model_plot_df[metric], width, label=metric)

plt.xticks(x + width * 2, model_plot_df["model"], rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Performance Comparison - V2 Cropped ROI")
plt.legend()
plt.tight_layout()

comparison_fig_path = os.path.join(FIGURES_DIR, "model_performance_comparison_v2.png")
plt.savefig(comparison_fig_path, dpi=300)
plt.close()

print(f"Figura guardada: {comparison_fig_path}")


# =========================
# 4. MATRIZ DE CONFUSIÓN KNN
# =========================

plt.figure(figsize=(9, 7))

plt.imshow(cm_df.values)
plt.title("Confusion Matrix - KNN V2")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")

plt.xticks(
    ticks=np.arange(len(cm_df.columns)),
    labels=[col.replace("Predicted_", "") for col in cm_df.columns],
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(cm_df.index)),
    labels=[idx.replace("Actual_", "") for idx in cm_df.index]
)

for i in range(cm_df.shape[0]):
    for j in range(cm_df.shape[1]):
        plt.text(j, i, cm_df.iloc[i, j], ha="center", va="center")

plt.colorbar(label="Number of samples")
plt.tight_layout()

cm_fig_path = os.path.join(FIGURES_DIR, "confusion_matrix_knn_v2.png")
plt.savefig(cm_fig_path, dpi=300)
plt.close()

print(f"Figura guardada: {cm_fig_path}")


# =========================
# 5. ÁREA POR CLASE
# =========================

area_plot_df = area_df.sort_values(by="area_km2", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(area_plot_df["class_name"], area_plot_df["area_km2"])
plt.ylabel("Area (km²)")
plt.xlabel("Class")
plt.title("Classified Area by Class - V2 Cropped ROI")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

area_fig_path = os.path.join(FIGURES_DIR, "area_by_class_km2_v2.png")
plt.savefig(area_fig_path, dpi=300)
plt.close()

print(f"Figura guardada: {area_fig_path}")


# =========================
# 6. TABLAS LIMPIAS
# =========================

summary_clean = model_df[[
    "model",
    "accuracy",
    "kappa",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]].copy()

summary_clean = summary_clean.round(4)

summary_clean_path = os.path.join(METRICS_DIR, "model_comparison_clean_v2.csv")
summary_clean.to_csv(summary_clean_path, index=False)

area_clean = area_df.copy()
area_clean["area_ha"] = area_clean["area_ha"].round(2)
area_clean["area_km2"] = area_clean["area_km2"].round(4)

area_clean_path = os.path.join(METRICS_DIR, "area_by_class_clean_v2.csv")
area_clean.to_csv(area_clean_path, index=False)

print(f"Tabla limpia guardada: {summary_clean_path}")
print(f"Tabla limpia de áreas guardada: {area_clean_path}")

print("\nProceso terminado correctamente.")