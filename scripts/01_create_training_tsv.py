import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from pyproj import Transformer


# =========================
# 1. RUTAS DEL PROYECTO
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BANDS_10M_DIR = os.path.join(BASE_DIR, "data", "bands_10m")
BANDS_20M_DIR = os.path.join(BASE_DIR, "data", "bands_20m")
POLYGONS_PATH = os.path.join(BASE_DIR, "data", "polygons", "training_polygons2.geojson")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "tsv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_TSV = os.path.join(OUTPUT_DIR, "training_data.tsv")


# =========================
# 2. BANDAS A USAR
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


# =========================
# 3. VALIDAR ARCHIVOS
# =========================

print("Verificando archivos...")

for band, path in band_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la banda {band}: {path}")

if not os.path.exists(POLYGONS_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de polígonos: {POLYGONS_PATH}")

print("Todos los archivos fueron encontrados.")


# =========================
# 4. LEER BANDA DE REFERENCIA 10 M
# =========================

reference_band = "B02"

with rasterio.open(band_paths[reference_band]) as ref:
    ref_profile = ref.profile
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = ref.shape
    ref_bounds = ref.bounds

print("CRS de referencia:", ref_crs)
print("Tamaño raster de referencia:", ref_shape)


# =========================
# 5. FUNCIÓN PARA LEER Y REMUESTREAR
# =========================

def read_band_to_reference_grid(path, ref_shape, ref_transform, ref_crs):
    """
    Lee una banda raster y la reproyecta/remuestrea a la grilla de referencia de 10 m.
    """
    with rasterio.open(path) as src:
        src_array = src.read(1)

        destination = np.empty(ref_shape, dtype=np.float32)

        reproject(
            source=src_array,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )

    return destination


# =========================
# 6. CARGAR TODAS LAS BANDAS
# =========================

print("Cargando y remuestreando bandas...")

band_arrays = {}

for band_name, band_path in band_paths.items():
    print(f"Procesando {band_name}...")
    band_arrays[band_name] = read_band_to_reference_grid(
        band_path,
        ref_shape,
        ref_transform,
        ref_crs
    )

print("Bandas cargadas correctamente.")


# =========================
# 7. LEER POLÍGONOS
# =========================

polygons = gpd.read_file(POLYGONS_PATH)

print("CRS polígonos:", polygons.crs)

if polygons.crs != ref_crs:
    print("Reproyectando polígonos al CRS del raster...")
    polygons = polygons.to_crs(ref_crs)

required_columns = ["class_id", "class_name", "geometry"]

for col in required_columns:
    if col not in polygons.columns:
        raise ValueError(f"Falta la columna requerida: {col}")

print("Número de polígonos por clase:")
print(polygons["class_name"].value_counts())


# =========================
# 8. TRANSFORMADOR A LAT/LON
# =========================

transformer = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)


# =========================
# 9. EXTRAER PÍXELES POR POLÍGONO
# =========================

records = []

print("Extrayendo píxeles de los polígonos...")

for idx, row in polygons.iterrows():
    geom = row.geometry
    class_id = row["class_id"]
    class_name = row["class_name"]

    mask = geometry_mask(
        [geom],
        transform=ref_transform,
        invert=True,
        out_shape=ref_shape
    )

    rows, cols = np.where(mask)

    for r, c in zip(rows, cols):
        x, y = rasterio.transform.xy(ref_transform, r, c, offset="center")
        lon, lat = transformer.transform(x, y)

        pixel_data = {
            "Latitude": lat,
            "Longitude": lon,
        }

        valid_pixel = True

        for band_name, array in band_arrays.items():
            value = array[r, c]

            if np.isnan(value):
                valid_pixel = False
                break

            pixel_data[band_name] = float(value)

        if valid_pixel:
            pixel_data["class_id"] = int(class_id)
            pixel_data["class_name"] = str(class_name)
            records.append(pixel_data)

print(f"Total de píxeles extraídos: {len(records)}")


# =========================
# 10. CREAR DATAFRAME Y EXPORTAR TSV
# =========================

df = pd.DataFrame(records)

columns_order = [
    "Latitude",
    "Longitude",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
    "class_id",
    "class_name",
]

df = df[columns_order]

print("Muestras por clase:")
print(df["class_name"].value_counts())

df.to_csv(OUTPUT_TSV, sep="\t", index=False)

print(f"Archivo TSV guardado en: {OUTPUT_TSV}")