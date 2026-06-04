import os
import rasterio
import geopandas as gpd
from rasterio.mask import mask


# =========================
# 1. RUTAS
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ROI_PATH = os.path.join(BASE_DIR, "data", "roi", "roi_hunga_tonga.geojson")

ORIGINAL_10M_DIR = os.path.join(BASE_DIR, "data", "bands_10m")
ORIGINAL_20M_DIR = os.path.join(BASE_DIR, "data", "bands_20m")

CROPPED_10M_DIR = os.path.join(BASE_DIR, "data", "cropped", "bands_10m")
CROPPED_20M_DIR = os.path.join(BASE_DIR, "data", "cropped", "bands_20m")

os.makedirs(CROPPED_10M_DIR, exist_ok=True)
os.makedirs(CROPPED_20M_DIR, exist_ok=True)


# =========================
# 2. BANDAS A RECORTAR
# =========================

bands = {
    "B02_10m.jp2": (ORIGINAL_10M_DIR, CROPPED_10M_DIR),
    "B03_10m.jp2": (ORIGINAL_10M_DIR, CROPPED_10M_DIR),
    "B04_10m.jp2": (ORIGINAL_10M_DIR, CROPPED_10M_DIR),
    "B08_10m.jp2": (ORIGINAL_10M_DIR, CROPPED_10M_DIR),

    "B05_20m.jp2": (ORIGINAL_20M_DIR, CROPPED_20M_DIR),
    "B06_20m.jp2": (ORIGINAL_20M_DIR, CROPPED_20M_DIR),
    "B07_20m.jp2": (ORIGINAL_20M_DIR, CROPPED_20M_DIR),
    "B8A_20m.jp2": (ORIGINAL_20M_DIR, CROPPED_20M_DIR),
    "B11_20m.jp2": (ORIGINAL_20M_DIR, CROPPED_20M_DIR),
    "B12_20m.jp2": (ORIGINAL_20M_DIR, CROPPED_20M_DIR),
}


# =========================
# 3. VALIDAR ROI
# =========================

if not os.path.exists(ROI_PATH):
    raise FileNotFoundError(f"No se encontró el ROI: {ROI_PATH}")

roi = gpd.read_file(ROI_PATH)

if roi.empty:
    raise ValueError("El archivo ROI está vacío.")

print("ROI cargado correctamente.")
print("CRS ROI:", roi.crs)


# =========================
# 4. RECORTAR BANDAS
# =========================

for filename, (input_dir, output_dir) in bands.items():
    input_path = os.path.join(input_dir, filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró la banda: {input_path}")

    output_name = filename.replace(".jp2", ".tif")
    output_path = os.path.join(output_dir, output_name)

    print(f"Recortando {filename}...")

    with rasterio.open(input_path) as src:
        roi_proj = roi.to_crs(src.crs)
        geometries = [geom for geom in roi_proj.geometry]

        clipped_array, clipped_transform = mask(
            src,
            geometries,
            crop=True,
            nodata=0
        )

        clipped_profile = src.profile.copy()
        clipped_profile.update({
            "driver": "GTiff",
            "height": clipped_array.shape[1],
            "width": clipped_array.shape[2],
            "transform": clipped_transform,
            "nodata": 0,
            "compress": "lzw"
        })

        with rasterio.open(output_path, "w", **clipped_profile) as dst:
            dst.write(clipped_array)

    print(f"Guardado: {output_path}")

print("\nRecorte terminado correctamente.")