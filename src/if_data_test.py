import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "Dataset archivos tiff"
METEO_DIR = BASE_DIR / "data" / "meteo"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Diccionario para identificar parcelas etiquetadas por su sufijo
MAPEO_ETIQUETADAS = {
    "": "Kalena_Parcela_1", "-2": "Kalena_Parcela_2", "-3": "Kalena_Parcela_3",
    "-4": "Kalena_Parcela_4", "-5": "Kalena_Parcela_5", "-6": "Kalena_Parcela_6",
    "-7": "Basilio_Parcela_7", "-8": "Basilio_Parcela_8", "-9": "Basilio_Parcela_9",
    "-10": "Basilio_Parcela_10", "-11": "Basilio_Parcela_11", "-12": "Basilio_Parcela_12",
    "-13": "Basilio_Parcela_13", "-14": "Basilio_Parcela_14", "-15": "Basilio_Parcela_15",
    "-16": "Basilio_Parcela_16",
    "-1-1 Los Helechos I": "HelechosI_Parcela_1_1", "-1-2 Los Helechos I": "HelechosI_Parcela_1_2",
    "-1-3 Los Helechos I": "HelechosI_Parcela_1_3", "-1-4 Los Helechos I": "HelechosI_Parcela_1_4",
    "-2-1 Los Helechos II": "HelechosII_Parcela_2_1", "-2-2 Los Helechos II": "HelechosII_Parcela_2_2",
    "-3-1 Los Helechos III": "HelechosIII_Parcela_3_1", "-3-2 Los Helechos III": "HelechosIII_Parcela_3_2",
    "-4-5 Jardín América": "Jardin_America_Parcela_4_5",
    "-6-1 Caraguatay": "Caraguatay_Parcela_6_1", "-6-2 Caraguatay": "Caraguatay_Parcela_6_2",
    "-6-3 Caraguatay": "Caraguatay_Parcela_6_3"
}

def mapear_estacion_meteorologica(parcela: str) -> str:
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
    archivos = list(ruta_carpeta.glob(f"*{patron}*.tiff"))
    if not archivos: return None
    with rasterio.open(archivos[0]) as src:
        return src.read(1).astype(np.float32)

def crear_conjunto_test_if():
    print("Iniciando creación del Conjunto de Test a NIVEL PÍXEL (Isolation Forest)...\n")
    
    # 1. Leer parcelas objetivo desde el rf_test.csv
    try:
        df_rf_test = pd.read_csv(PROCESSED_DIR / "rf_test.csv")
        parcelas_test = df_rf_test['parcela'].unique().tolist()
        print(f"Parcelas a extraer para el examen: {parcelas_test}\n")
    except FileNotFoundError:
        print("Error: No se encontró rf_test.csv. Debe particionar los datos de RF primero.")
        return

    registros_pixeles = []
    
    # 2. Escanear SOLO la carpeta de Etiquetadas
    carpetas_fecha = set([ruta.parent for ruta in RAW_DIR.rglob("*.tiff") if "02_Etiquetadas" in ruta.parts])
    
    for ruta_fecha in carpetas_fecha:
        partes = ruta_fecha.parts
        idx_base = partes.index("Dataset archivos tiff")
        subruta = partes[idx_base+1:]
        
        etiqueta = subruta[1]
        nombre_carpeta_fecha = subruta[-1]
        fecha_real = nombre_carpeta_fecha[:10]
        sufijo = nombre_carpeta_fecha[10:]
        parcela = MAPEO_ETIQUETADAS.get(sufijo, f"Desconocida ({sufijo})")
        
        # Filtro vital: Si esta parcela etiquetada NO es parte del examen, la ignoramos
        if parcela not in parcelas_test:
            continue
            
        ubicacion = mapear_estacion_meteorologica(parcela)
        
        b02 = leer_banda(ruta_fecha, "B02")
        b04 = leer_banda(ruta_fecha, "B04")
        b05 = leer_banda(ruta_fecha, "B05")
        b08 = leer_banda(ruta_fecha, "B08")
        b11 = leer_banda(ruta_fecha, "B11")
        b12 = leer_banda(ruta_fecha, "B12")
        mascara = leer_banda(ruta_fecha, "mask")
        
        if b04 is None or b08 is None: continue
            
        if mascara is not None:
            condicion_valida = mascara == 1
            b02 = np.where(condicion_valida, b02, np.nan) if b02 is not None else None
            b04 = np.where(condicion_valida, b04, np.nan)
            b05 = np.where(condicion_valida, b05, np.nan) if b05 is not None else None
            b08 = np.where(condicion_valida, b08, np.nan)
            b11 = np.where(condicion_valida, b11, np.nan) if b11 is not None else None
            b12 = np.where(condicion_valida, b12, np.nan) if b12 is not None else None

        def calc_indice(num, den):
            return np.divide(num, den, out=np.full_like(den, np.nan), where=(den!=0))

        ndvi = calc_indice(b08 - b04, b08 + b04)
        ndvi_flat = ndvi.flatten()
        mascara_veg = (~np.isnan(ndvi_flat)) & (ndvi_flat >= 0.0)
        
        if not np.any(mascara_veg): continue
            
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
            'ubicacion_meteo': ubicacion,
            'etiqueta': etiqueta # GUARDAMOS LA ETIQUETA REAL PARA CORREGIR EL EXAMEN LUEGO
        }
        registros_pixeles.append(pd.DataFrame(datos))
        print(f"Extraídos {np.sum(mascara_veg)} píxeles de {parcela} ({etiqueta})")

    df_pixeles = pd.concat(registros_pixeles, ignore_index=True)
    df_pixeles['fecha'] = pd.to_datetime(df_pixeles['fecha'], format='%d-%m-%Y', errors='coerce')
    
    # 3. Fusión Climática
    lista_meteo = []
    estaciones = df_pixeles['ubicacion_meteo'].dropna().unique()
    for est in estaciones:
        ruta_csv = METEO_DIR / f"datos_meteorologicos_{est}.csv"
        if ruta_csv.exists():
            df_t = pd.read_csv(ruta_csv)
            df_t['ubicacion_meteo'] = est
            lista_meteo.append(df_t)
            
    df_meteo = pd.concat(lista_meteo, ignore_index=True)
    df_meteo['fecha'] = pd.to_datetime(df_meteo['fecha'])
    df_meteo = df_meteo.sort_values(by=['ubicacion_meteo', 'fecha']).set_index('fecha')
    
    df_clima_30d = df_meteo.groupby('ubicacion_meteo').rolling('30D').agg({
        'temp_max': 'mean', 'temp_min': 'mean', 'precipitacion_acumulada': 'sum',
        'humedad_promedio': 'mean', 'viento_max': 'max'
    }).reset_index()
    
    df_clima_30d.rename(columns={'temp_max': 'temp_max_30d_prom', 'temp_min': 'temp_min_30d_prom',
                                 'precipitacion_acumulada': 'precip_acumulada_30d', 
                                 'humedad_promedio': 'humedad_30d_prom', 'viento_max': 'viento_max_30d'}, inplace=True)
    
    # 4. Cruce, limpieza y guardado
    dataset_final = pd.merge(df_pixeles, df_clima_30d, on=['ubicacion_meteo', 'fecha'], how='inner')
    dataset_final = dataset_final.dropna()
    
    indices_simetricos = ['NDVI', 'EVI', 'SAVI', 'NDRE', 'NDMI', 'NBR']
    for ind in indices_simetricos:
        dataset_final = dataset_final[(dataset_final[ind] >= -1.0) & (dataset_final[ind] <= 1.0)]
    
    indices_positivos = ['MTCI', 'MSI']
    for ind in indices_positivos:
        dataset_final = dataset_final[dataset_final[ind] >= 0.0]
    
    ruta_salida = PROCESSED_DIR / "if_test.csv"
    dataset_final.to_csv(ruta_salida, index=False)
    print(f"\n¡Dataset de TEST a nivel píxel guardado en: {ruta_salida}!")
    print(f"Total de píxeles listos para el Paso 5: {len(dataset_final)}")

if __name__ == "__main__":
    crear_conjunto_test_if()