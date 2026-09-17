import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import warnings
# Ocultamos las advertencias matemáticas de Numpy. Al dividir matrices de píxeles, 
# es normal que haya divisiones por cero (ej. píxeles vacíos), las cuales manejamos de forma segura internamente.
warnings.filterwarnings('ignore') 

# ==========================================
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# ==========================================
# Resolvemos la ruta absoluta para garantizar que el script funcione en cualquier computadora sin fallar
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "Dataset archivos tiff"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def mapear_estacion_meteorologica(parcela: str) -> str:
    """
    Función traductora: Vincula el nombre de la parcela física con el archivo 
    de la estación meteorológica más cercana que le corresponde.
    """
    parcela_upper = str(parcela).upper()
    if "BASILIO" in parcela_upper: return "BASILIO"
    elif "CARAGUATAY" in parcela_upper: return "CARAGUATAY"
    elif "JARDIN_AMERICA" in parcela_upper: return "JARDIN_AMERICA"
    elif "KALENA" in parcela_upper: return "KALENA"
    elif "HELECHOSI_" in parcela_upper: return "LOS_HELECHOS_1"
    elif "HELECHOSII_" in parcela_upper: return "LOS_HELECHOS_2"
    elif "HELECHOSIII_" in parcela_upper: return "LOS_HELECHOS_3"
    else: return None

def leer_banda(ruta_carpeta: Path, patron: str) -> np.ndarray:
    """
    Busca un archivo TIFF específico dentro de una carpeta y extrae su matriz de datos.
    Devuelve la imagen convertida en un array numérico de 32 bits (alta precisión).
    """
    archivos = list(ruta_carpeta.glob(f"*{patron}*.tiff"))
    if not archivos: return None
    with rasterio.open(archivos[0]) as src:
        return src.read(1).astype(np.float32)

def extraer_pixeles_if():
    """
    Script principal para la extracción granular de datos.
    Desarma las imágenes de las parcelas no etiquetadas en píxeles individuales (resolución nativa),
    aplica filtros de vegetación (NDVI >= 0) y los cruza con datos climáticos.
    """
    print("IF Fase 1: Extracción a NIVEL PÍXEL...\n")
    registros_pixeles = []

    # ==========================================
    # 1. ESCANEO SOLO DE IMÁGENES NO ETIQUETADAS
    # ==========================================
    # El Isolation Forest es no supervisado, por lo que solo debe ver el lote de datos 
    # de "monitoreo general" para aprender cómo es un suelo normal.
    rutas_no_etiquetadas = list(RAW_DIR.rglob("01_No_Etiquetadas/*"))
    carpetas_fecha = set([ruta.parent for ruta in RAW_DIR.rglob("*.tiff") if "01_No_Etiquetadas" in ruta.parts])

    # ==========================================
    # BLINDAJE ESPACIAL (Prevención de Data Leakage)
    # ==========================================
    # Leemos el archivo de Test generado por el Random Forest para saber 
    # qué parcelas están estrictamente reservadas para el examen final.
    try:
        df_test_rf = pd.read_csv(PROCESSED_DIR / "rf_test.csv")
        parcelas_prohibidas = df_test_rf['parcela'].unique().tolist()
        print(f"Parcelas bloqueadas para el entrenamiento: {parcelas_prohibidas}\n")
    except FileNotFoundError:
        print("Error: Falta rf_test.csv. Ejecute data_splitting.py primero.")
        return
    
    for ruta_fecha in carpetas_fecha:
        # Extracción de metadatos desde el nombre de la carpeta
        nombre_carpeta = ruta_fecha.name
        fecha_real = nombre_carpeta[:10]
        parcela = ruta_fecha.parent.name

        parcela = ruta_fecha.parent.name
        
        # Si la parcela está en la lista de prohibidas (examen final), la ignoramos
        if parcela in parcelas_prohibidas:
            continue
            
        ubicacion = mapear_estacion_meteorologica(parcela)

        ubicacion = mapear_estacion_meteorologica(parcela)

        # Lectura de las bandas espectrales físicas capturadas por Sentinel-2
        b02 = leer_banda(ruta_fecha, "B02") # Azul
        b04 = leer_banda(ruta_fecha, "B04") # Rojo
        b05 = leer_banda(ruta_fecha, "B05") # Borde Rojo
        b08 = leer_banda(ruta_fecha, "B08") # Infrarrojo Cercano
        b11 = leer_banda(ruta_fecha, "B11") # Infrarrojo de Onda Corta (SWIR 1)
        b12 = leer_banda(ruta_fecha, "B12") # Infrarrojo de Onda Corta (SWIR 2)
        mascara = leer_banda(ruta_fecha, "mask") # Máscara de calidad de píxel
        
        # Si faltan las bandas esenciales para calcular vigor (Rojo o NIR), descartamos la imagen
        if b04 is None or b08 is None: continue

        # ==========================================
        # FILTRADO DE NUBES (Cloud Masking)
        # ==========================================
        if mascara is not None:
            # Conservamos únicamente los píxeles marcados como '1' (cielo despejado/válido).
            # Reemplazamos las nubes, sombras o errores con np.nan (nulo) para que no alteren la matemática.
            condicion_valida = mascara == 1
            b02 = np.where(condicion_valida, b02, np.nan) if b02 is not None else None
            b04 = np.where(condicion_valida, b04, np.nan)
            b05 = np.where(condicion_valida, b05, np.nan) if b05 is not None else None
            b08 = np.where(condicion_valida, b08, np.nan)
            b11 = np.where(condicion_valida, b11, np.nan) if b11 is not None else None
            b12 = np.where(condicion_valida, b12, np.nan) if b12 is not None else None

        # ==========================================
        # CÁLCULO DE ÍNDICES A NIVEL PÍXEL
        # ==========================================
        def calc_indice(num, den):
            """Divide dos matrices. Si el denominador es 0, asigna np.nan de forma segura."""
            return np.divide(num, den, out=np.full_like(den, np.nan), where=(den!=0))

        # Calculamos el NDVI primero porque lo usaremos como "Filtro Biológico" principal
        ndvi = calc_indice(b08 - b04, b08 + b04)

        # EL FILTRO MÁGICO DE VEGETACIÓN (Prevención de Falsos Positivos)
        # 1. flatten(): Transforma la matriz 2D de la foto en un vector (lista) 1D de píxeles individuales.
        # 2. mascara_veg: Exigimos que el píxel NO sea nulo y que su NDVI sea >= 0.0
        # Esto elimina matemáticamente el agua, sombras profundas, nubes residuales y asfalto, 
        # garantizando que al modelo solo entren píxeles de suelo o vegetación real.
        ndvi_flat = ndvi.flatten()
        mascara_veg = (~np.isnan(ndvi_flat)) & (ndvi_flat >= 0.0)

        # Si la imagen era pura nube o no quedó ningún píxel válido, abortamos y seguimos con la próxima
        if not np.any(mascara_veg): continue

        # Generamos el diccionario extrayendo SOLAMENTE los píxeles que pasaron el filtro (mascara_veg).
        # Cada índice espectral se calcula, se aplana y se filtra al vuelo para evitar colapsar la memoria RAM.
        datos = {
            'NDVI': ndvi_flat[mascara_veg],
            'SAVI': calc_indice((b08 - b04) * 1.5, b08 + b04 + 0.5).flatten()[mascara_veg],
            'EVI': calc_indice(2.5 * (b08 - b04), b08 + 6 * b04 - 7.5 * b02 + 1).flatten()[mascara_veg] if b02 is not None else np.nan,
            'NDRE': calc_indice(b08 - b05, b08 + b05).flatten()[mascara_veg] if b05 is not None else np.nan,
            'MTCI': calc_indice(b08 - b05, b05 - b04).flatten()[mascara_veg] if b05 is not None else np.nan,
            'NDMI': calc_indice(b08 - b11, b08 + b11).flatten()[mascara_veg] if b11 is not None else np.nan,
            'MSI': calc_indice(b11, b08).flatten()[mascara_veg] if b11 is not None else np.nan,
            'NBR': calc_indice(b08 - b12, b08 + b12).flatten()[mascara_veg] if b12 is not None else np.nan,
            'fecha': fecha_real,
            'parcela': parcela,
            'ubicacion_meteo': ubicacion
        }
        # Agregamos los miles de píxeles de esta foto a nuestra lista maestra
        registros_pixeles.append(pd.DataFrame(datos))
        print(f"Extraídos {np.sum(mascara_veg)} píxeles de {parcela} ({fecha_real})")

    # Unimos todos los recuadros de datos en un solo DataFrame gigante
    df_pixeles = pd.concat(registros_pixeles, ignore_index=True)
    df_pixeles['fecha'] = pd.to_datetime(df_pixeles['fecha'], format='%d-%m-%Y', errors='coerce')

    #Finalmente guardamos el dataset de píxeles extraídos en un CSV temporal para la siguiente fase de fusión con datos climáticos
    ruta_salida = PROCESSED_DIR / "if_temp_pixels.csv"
    df_pixeles.to_csv(ruta_salida, index=False)
    print(f"\nTotal de píxeles extraídos y guardados en CSV temporal: {len(df_pixeles)}")

if __name__ == "__main__":
    extraer_pixeles_if()