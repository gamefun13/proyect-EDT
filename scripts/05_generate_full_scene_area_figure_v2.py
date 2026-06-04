import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. RUTAS
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

METRICS_DIR = os.path.join(BASE_DIR, "outputs_v2", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs_v2", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

FULL_SCENE_AREA_PATH = os.path.join(
    METRICS_DIR,
    "area_by_class_full_scene_v2_model.csv"
)

ROI_AREA_PATH = os.path.join(
    METRICS_DIR,
    "area_by_class_v2.csv"
)

OUTPUT_FULL_SCENE_FIG = os.path.join(
    FIGURES_DIR,
    "area_by_class_full_scene_v2_model.png"
)

OUTPUT_COMPARISON_FIG = os.path.join(
    FIGURES_DIR,
    "area_comparison_roi_vs_full_scene_v2.png"
)

OUTPUT_CLEAN_CSV = os.path.join(
    METRICS_DIR,
    "area_by_class_full_scene_v2_model_clean.csv"
)


# =========================
# 2. CARGAR ÁREAS ESCENA COMPLETA
# =========================

if not os.path.exists(FULL_SCENE_AREA_PATH):
    raise FileNotFoundError(f"No se encontró el archivo: {FULL_SCENE_AREA_PATH}")

full_df = pd.read_csv(FULL_SCENE_AREA_PATH)

full_df = full_df.sort_values(by="area_km2", ascending=False)

print("Áreas escena completa con modelo v2:")
print(full_df)


# =========================
# 3. GUARDAR TABLA LIMPIA
# =========================

clean_df = full_df.copy()
clean_df["area_ha"] = clean_df["area_ha"].round(2)
clean_df["area_km2"] = clean_df["area_km2"].round(4)

clean_df.to_csv(OUTPUT_CLEAN_CSV, index=False)

print(f"Tabla limpia guardada en: {OUTPUT_CLEAN_CSV}")


# =========================
# 4. GRÁFICA ÁREA ESCENA COMPLETA
# =========================

plt.figure(figsize=(10, 6))

plt.bar(full_df["class_name"], full_df["area_km2"])

plt.title("Classified Area by Class - Full Scene with V2 Model")
plt.xlabel("Class")
plt.ylabel("Area (km²)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(OUTPUT_FULL_SCENE_FIG, dpi=300)
plt.close()

print(f"Figura guardada en: {OUTPUT_FULL_SCENE_FIG}")


# =========================
# 5. GRÁFICA COMPARATIVA ROI VS ESCENA COMPLETA
# =========================

if os.path.exists(ROI_AREA_PATH):
    roi_df = pd.read_csv(ROI_AREA_PATH)

    roi_df = roi_df[["class_name", "area_km2"]].rename(
        columns={"area_km2": "ROI_km2"}
    )

    full_comp_df = full_df[["class_name", "area_km2"]].rename(
        columns={"area_km2": "Full_scene_km2"}
    )

    comparison_df = pd.merge(
        full_comp_df,
        roi_df,
        on="class_name",
        how="outer"
    ).fillna(0)

    comparison_df = comparison_df.sort_values(
        by="Full_scene_km2",
        ascending=False
    )

    x = range(len(comparison_df))

    plt.figure(figsize=(11, 6))

    plt.bar(
        [i - 0.2 for i in x],
        comparison_df["Full_scene_km2"],
        width=0.4,
        label="Full scene"
    )

    plt.bar(
        [i + 0.2 for i in x],
        comparison_df["ROI_km2"],
        width=0.4,
        label="Cropped ROI"
    )

    plt.title("Area Comparison: Cropped ROI vs Full Scene - V2 Model")
    plt.xlabel("Class")
    plt.ylabel("Area (km²)")
    plt.xticks(x, comparison_df["class_name"], rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_COMPARISON_FIG, dpi=300)
    plt.close()

    print(f"Figura comparativa guardada en: {OUTPUT_COMPARISON_FIG}")

else:
    print("No se encontró area_by_class_v2.csv. Se omitió la figura comparativa.")


print("\nProceso terminado correctamente.")