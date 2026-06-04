import os
import joblib
import numpy as np
import pandas as pd
import rasterio

from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window


# =========================
# 1. RUTAS
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Bandas originales completas
BANDS_10M_DIR = os.path.join(BASE_DIR, "data", "bands_10m")
BANDS_20M_DIR = os.path.join(BASE_DIR, "data", "bands_20m")

# Modelo v2 entrenado con ROI recortado y 6 clases
MODELS_DIR = os.path.join(BASE_DIR, "outputs_v2", "models")

# Salidas dentro de outputs_v2
RASTERS_DIR = os.path.join(BASE_DIR, "outputs_v2", "rasters")
METRICS_DIR = os.path.join(BASE_DIR, "outputs_v2", "metrics")

os.makedirs(RASTERS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

OUTPUT_RASTER = os.path.join(
    RASTERS_DIR,
    "classified_hunga_tonga_full_scene_v2_model.tif"
)

OUTPUT_AREA_CSV = os.path.join(
    METRICS_DIR,
    "area_by_class_full_scene_v2_model.csv"
)

OUTPUT_MAPPING_CSV = os.path.join(
    METRICS_DIR,
    "class_mapping_full_scene_v2_model.csv"
)


# =========================
# 2. BANDAS ORIGINALES COMPLETAS
# =========================

band_paths = {
    "B02": os.path.join(BANDS_10M_DIR, "B02_10m.jp2"),
    "B03": os.path.join(BANDS_10M_DIR, "B03_10m.jp2"),
    "B04": os.path.join(BANDS_10M_DIR, "B04_10m.jp2"),
    "B08": os.path.join(BANDS_10M_DIR, "B08_10m.jp2"),

    "B05": os.path.join(BANDS_20M_DIR, "B05_20m.jp2"),
    "B06": os.path.join(BANDS_20M_DIR, "B06_20m.jp2"),
    "B07": os.path.join(BANDS_20M_DIR, "B07_20m.jp2"),
    "B8A": os.path.join(BANDS_20M_DIR, "B8A_20m.jp2"),
    "B11": os.path.join(BANDS_20M_DIR, "B11_20m.jp2"),
    "B12": os.path.join(BANDS_20M_DIR, "B12_20m.jp2"),
}

feature_columns = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]


# =========================
# 3. VALIDAR ARCHIVOS
# =========================

print("Verificando archivos...")

for band, path in band_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la banda {band}: {path}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo v2: {MODEL_PATH}")

if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"No se encontró el label encoder v2: {LABEL_ENCODER_PATH}")

print("Todos los archivos fueron encontrados.")


# =========================
# 4. CARGAR MODELO V2
# =========================

print("Cargando modelo v2 y label encoder...")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

class_names = list(label_encoder.classes_)

print("Clases del modelo v2:")
for idx, name in enumerate(class_names):
    print(f"Código raster {idx + 1} -> {name}")

mapping_df = pd.DataFrame({
    "raster_code": list(range(1, len(class_names) + 1)),
    "model_label": list(range(len(class_names))),
    "class_name": class_names
})

mapping_df.to_csv(OUTPUT_MAPPING_CSV, index=False)
print(f"Mapeo de clases guardado en: {OUTPUT_MAPPING_CSV}")


# =========================
# 5. RASTER DE REFERENCIA COMPLETO
# =========================

reference_path = band_paths["B02"]

with rasterio.open(reference_path) as ref:
    ref_profile = ref.profile.copy()
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_width = ref.width
    ref_height = ref.height

print("CRS:", ref_crs)
print("Tamaño:", ref_width, "x", ref_height)
print("Transform:", ref_transform)


# =========================
# 6. PERFIL DEL RASTER FINAL
# =========================

output_profile = ref_profile.copy()
output_profile.update({
    "driver": "GTiff",
    "count": 1,
    "dtype": "uint8",
    "nodata": 0,
    "compress": "lzw"
})


# =========================
# 7. ABRIR BANDAS CON WARPEDVRT
# =========================

src_datasets = {}
vrt_datasets = {}

try:
    for band_name, path in band_paths.items():
        src = rasterio.open(path)
        src_datasets[band_name] = src

        vrt = WarpedVRT(
            src,
            crs=ref_crs,
            transform=ref_transform,
            width=ref_width,
            height=ref_height,
            resampling=Resampling.bilinear
        )

        vrt_datasets[band_name] = vrt

    # =========================
    # 8. PREDICCIÓN POR BLOQUES
    # =========================

    print("Clasificando escena completa con el modelo v2...")

    class_pixel_counts = {
        class_name: 0 for class_name in class_names
    }

    block_size = 512

    with rasterio.open(OUTPUT_RASTER, "w", **output_profile) as dst:
        total_blocks = 0

        for row_off in range(0, ref_height, block_size):
            for col_off in range(0, ref_width, block_size):

                height = min(block_size, ref_height - row_off)
                width = min(block_size, ref_width - col_off)

                window = Window(col_off, row_off, width, height)

                band_stack = []

                for band_name in feature_columns:
                    arr = vrt_datasets[band_name].read(1, window=window).astype(np.float32)
                    band_stack.append(arr)

                stack = np.stack(band_stack, axis=-1)

                flat_pixels = stack.reshape(-1, len(feature_columns))

                valid_mask = (
                    np.isfinite(flat_pixels).all(axis=1) &
                    (flat_pixels > 0).all(axis=1)
                )

                prediction_flat = np.zeros(flat_pixels.shape[0], dtype=np.uint8)

                if valid_mask.sum() > 0:
                    valid_pixels = flat_pixels[valid_mask]

                    # Mantener nombres de columnas para evitar warnings de sklearn
                    valid_pixels_df = pd.DataFrame(valid_pixels, columns=feature_columns)

                    pred_labels = model.predict(valid_pixels_df)

                    # Modelo predice 0,1,2,3,4,5.
                    # Raster guarda 1,2,3,4,5,6.
                    pred_codes = pred_labels.astype(np.uint8) + 1

                    prediction_flat[valid_mask] = pred_codes

                    unique_codes, counts = np.unique(pred_codes, return_counts=True)

                    for code, count in zip(unique_codes, counts):
                        class_name = class_names[int(code) - 1]
                        class_pixel_counts[class_name] += int(count)

                prediction_block = prediction_flat.reshape(height, width)

                dst.write(prediction_block, 1, window=window)

                total_blocks += 1

                if total_blocks % 20 == 0:
                    print(f"Bloques procesados: {total_blocks}")

    print("Raster completo clasificado con modelo v2 guardado en:")
    print(OUTPUT_RASTER)

finally:
    for vrt in vrt_datasets.values():
        vrt.close()

    for src in src_datasets.values():
        src.close()


# =========================
# 9. CÁLCULO DE ÁREAS
# =========================

print("Calculando áreas por clase...")

pixel_width = abs(ref_transform.a)
pixel_height = abs(ref_transform.e)
pixel_area_m2 = pixel_width * pixel_height

area_records = []

for class_name, pixel_count in class_pixel_counts.items():
    area_m2 = pixel_count * pixel_area_m2
    area_ha = area_m2 / 10000
    area_km2 = area_m2 / 1_000_000

    area_records.append({
        "class_name": class_name,
        "pixel_count": pixel_count,
        "area_m2": area_m2,
        "area_ha": area_ha,
        "area_km2": area_km2
    })

area_df = pd.DataFrame(area_records)
area_df = area_df.sort_values(by="area_ha", ascending=False)

area_df.to_csv(OUTPUT_AREA_CSV, index=False)

print("Áreas por clase:")
print(area_df)

print(f"Tabla de áreas guardada en: {OUTPUT_AREA_CSV}")

print("\nProceso terminado correctamente.")