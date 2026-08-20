import os
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "Dataset archivos tiff"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def leer_banda(ruta_carpeta: Path, patron: str) -> np.ndarray:
    """Busca un archivo TIFF por un patrón de texto en el nombre."""
    archivos = list(ruta_carpeta.glob(f"*{patron}*.tiff"))
    if not archivos:
        return None
    
    with rasterio.open(archivos[0]) as src:
        return src.read(1).astype(np.float32)

def procesar_carpeta_fecha(ruta_fecha: Path) -> dict:
    """Lee las bandas de una fecha específica, aplica la máscara (si existe) y calcula los índices."""
    b02 = leer_banda(ruta_fecha, "B02")
    b04 = leer_banda(ruta_fecha, "B04")
    b05 = leer_banda(ruta_fecha, "B05")
    b08 = leer_banda(ruta_fecha, "B08")
    b11 = leer_banda(ruta_fecha, "B11")
    b12 = leer_banda(ruta_fecha, "B12")
    
    mascara = leer_banda(ruta_fecha, "mask")
    
    # Si faltan las bandas críticas, no podemos calcular nada
    if b04 is None or b08 is None:
        return None

    # Si hay máscara, eliminamos las nubes. Si no hay (como en tus etiquetadas), esto se ignora.
    if mascara is not None:
        condicion_valida = mascara == 1
        b02 = np.where(condicion_valida, b02, np.nan) if b02 is not None else None
        b04 = np.where(condicion_valida, b04, np.nan)
        b05 = np.where(condicion_valida, b05, np.nan) if b05 is not None else None
        b08 = np.where(condicion_valida, b08, np.nan)
        b11 = np.where(condicion_valida, b11, np.nan) if b11 is not None else None
        b12 = np.where(condicion_valida, b12, np.nan) if b12 is not None else None
    
    epsilon = 1e-10
    L = 0.5 
    
    resultados = {}
    
    ndvi = (b08 - b04) / (b08 + b04 + epsilon)
    resultados['NDVI'] = np.nanmedian(ndvi)
    
    if b02 is not None:
        evi = 2.5 * ((b08 - b04) / (b08 + 6 * b04 - 7.5 * b02 + 1 + epsilon))
        resultados['EVI'] = np.nanmedian(evi)
        
    savi = ((b08 - b04) / (b08 + b04 + L + epsilon)) * (1 + L)
    resultados['SAVI'] = np.nanmedian(savi)
    
    if b05 is not None:
        ndre = (b08 - b05) / (b08 + b05 + epsilon)
        resultados['NDRE'] = np.nanmedian(ndre)
        
        mtci = (b08 - b05) / (b05 - b04 + epsilon)
        resultados['MTCI'] = np.nanmedian(mtci)
        
    if b11 is not None:
        ndwi_ndmi = (b08 - b11) / (b08 + b11 + epsilon)
        resultados['NDWI'] = np.nanmedian(ndwi_ndmi)
        resultados['NDMI'] = np.nanmedian(ndwi_ndmi)
        
        msi = b11 / (b08 + epsilon)
        resultados['MSI'] = np.nanmedian(msi)
        
    if b12 is not None:
        nbr = (b08 - b12) / (b08 + b12 + epsilon)
        resultados['NBR'] = np.nanmedian(nbr)
        
    return resultados

def extraer_datos_satelitales(directorio_base: Path):
    registros = []
    print(f"Iniciando escaneo inteligente de imágenes en: {directorio_base}...\n")
    
    # 1. rglob busca TODOS los archivos .tiff sin importar qué tan profundos estén
    archivos_tiff = list(directorio_base.rglob("*.tiff"))
    
    # 2. Obtenemos solo las carpetas únicas que contienen esos archivos
    carpetas_con_imagenes = set([archivo.parent for archivo in archivos_tiff])
    
    for ruta_fecha in carpetas_con_imagenes:
        partes = ruta_fecha.parts
        
        try:
            # Buscamos dónde empieza nuestra estructura de interés
            idx_base = partes.index("Dataset archivos tiff")
            subruta = partes[idx_base+1:] 
        except ValueError:
            continue
        
        # La fecha siempre es la última carpeta
        fecha = subruta[-1]
        
        # Deducción dinámica de etiquetas y parcelas
        if subruta[0] == "01_No_Etiquetadas":
            etiqueta = "No_Etiquetada"
            parcela = subruta[-2] if len(subruta) >= 3 else "Desconocida"
        elif subruta[0] == "02_Etiquetadas":
            etiqueta = subruta[1] if len(subruta) >= 2 else "Etiquetada_Generica"
            parcela = subruta[-2] if len(subruta) >= 4 else "Desconocida"
        else:
            continue
        
        print(f"Procesando: {parcela} | {fecha} | [{etiqueta}]")
        indices = procesar_carpeta_fecha(ruta_fecha)
        
        if indices:
            indices.update({"fecha": fecha, "parcela": parcela, "etiqueta": etiqueta})
            registros.append(indices)

    df = pd.DataFrame(registros)
    if not df.empty:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Parseo robusto de fechas (soporta tanto 01-05-2025 como 2025-05-01)
        df['fecha'] = pd.to_datetime(df['fecha'], format='mixed', dayfirst=True, errors='coerce')
        ruta_salida = PROCESSED_DIR / "indices_satelitales.csv"
        df.to_csv(ruta_salida, index=False)
        print(f"\n¡Procesamiento finalizado! Dataset guardado en: {ruta_salida} con {len(df)} registros.")
    else:
        print("\nNo se encontraron imágenes válidas para procesar.")

if __name__ == "__main__":
    extraer_datos_satelitales(RAW_DIR)