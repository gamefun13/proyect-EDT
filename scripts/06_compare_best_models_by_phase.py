import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# 1. RUTAS
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FIGURES_DIR = os.path.join(BASE_DIR, "outputs_v2", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

OUTPUT_FIGURE = os.path.join(
    FIGURES_DIR,
    "best_models_metrics_comparison_by_phase.png"
)


# =========================
# 2. MÉTRICAS DE LOS MEJORES MODELOS
# =========================

data = {
    "Metric": [
        "Accuracy", "Kappa", "Precision", "Recall", "F1-score",
        "Accuracy", "Kappa", "Precision", "Recall", "F1-score"
    ],
    "Score": [
        # Phase 1 - Initial classification, 5 classes, best model KNN
        0.9967, 0.9959, 0.9968, 0.9967, 0.9967,

        # Phase 2 - Improved classification, 6 classes, best model KNN
        0.9925, 0.9910, 0.9926, 0.9925, 0.9925
    ],
    "Model / Phase": [
        "KNN - Initial phase",
        "KNN - Initial phase",
        "KNN - Initial phase",
        "KNN - Initial phase",
        "KNN - Initial phase",

        "KNN - Improved phase",
        "KNN - Improved phase",
        "KNN - Improved phase",
        "KNN - Improved phase",
        "KNN - Improved phase",
    ]
}

df = pd.DataFrame(data)


# =========================
# 3. GRÁFICA
# =========================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=df,
    x="Metric",
    y="Score",
    hue="Model / Phase",
    palette=["blue", "green"]
)

plt.title("Comparison of Best Models by Phase")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0.98, 1.0)

for container in ax.containers:
    ax.bar_label(container, fmt="%.4f", fontsize=9, padding=3)

plt.legend(title="Model / Phase")
plt.tight_layout()

plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.show()

print(f"Figure saved at: {OUTPUT_FIGURE}")


# ============================================================
# 4. COMPARACIÓN DE ACCURACY DE LOS 5 MODELOS EN AMBAS FASES
# ============================================================

OUTPUT_ACCURACY_COMPARISON = os.path.join(
    FIGURES_DIR,
    "accuracy_comparison_all_models_by_phase.png"
)

accuracy_data = {
    "Model": [
        "KNN", "Decision Tree", "SVM", "ANN", "Naive Bayes",
        "KNN", "Decision Tree", "SVM", "ANN", "Naive Bayes"
    ],
    "Accuracy": [
        # Phase 1 - Initial classification, 5 classes
        0.9967, 0.9941, 0.9853, 0.9847, 0.8814,

        # Phase 2 - Improved classification, 6 classes
        0.9925, 0.9833, 0.9786, 0.9840, 0.8440
    ],
    "Phase": [
        "Initial phase - 5 classes",
        "Initial phase - 5 classes",
        "Initial phase - 5 classes",
        "Initial phase - 5 classes",
        "Initial phase - 5 classes",

        "Improved phase - 6 classes",
        "Improved phase - 6 classes",
        "Improved phase - 6 classes",
        "Improved phase - 6 classes",
        "Improved phase - 6 classes"
    ]
}

accuracy_df = pd.DataFrame(accuracy_data)

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=accuracy_df,
    x="Model",
    y="Accuracy",
    hue="Phase",
    palette=["blue", "green"]
)

plt.title("Accuracy Comparison of the Five Models in Both Phases")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0.80, 1.01)

for container in ax.containers:
    ax.bar_label(container, fmt="%.4f", fontsize=9, padding=3)

plt.legend(title="Classification phase")
plt.tight_layout()

plt.savefig(OUTPUT_ACCURACY_COMPARISON, dpi=300)
plt.show()

print(f"Accuracy comparison figure saved at: {OUTPUT_ACCURACY_COMPARISON}")