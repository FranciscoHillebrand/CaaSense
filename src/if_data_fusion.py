import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# ==========================================
# Resolvemos la ruta absoluta para garantizar que el script funcione en cualquier computadora sin fallar
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
METEO_DIR = BASE_DIR / "data" / "meteo"

def fusionar_clima_if():
    print("IF Fase 2: Fusión de Clima a Nivel Píxel...\n")
    # ==========================================
    # 1. CARGA DEL DATASET DE PÍXELES EXTRAÍDOS
    # ==========================================
    ruta_entrada = PROCESSED_DIR / "if_temp_pixels.csv"
    df_pixeles = pd.read_csv(ruta_entrada)
    df_pixeles['fecha'] = pd.to_datetime(df_pixeles['fecha'])

    # ==========================================
    # 2. FUSIÓN CON DATOS METEOROLÓGICOS HISTÓRICOS
    # ==========================================
    # Cargamos los archivos CSV de clima solo para las estaciones que están presentes en los píxeles extraídos
    lista_meteo = []
    estaciones = df_pixeles['ubicacion_meteo'].dropna().unique()
    for est in estaciones:
        ruta_csv = METEO_DIR / f"datos_meteorologicos_{est}.csv"
        if ruta_csv.exists():
            df_t = pd.read_csv(ruta_csv)
            df_t['ubicacion_meteo'] = est
            lista_meteo.append(df_t)

    # Unificamos los datos climáticos y los ordenamos cronológicamente        
    df_meteo = pd.concat(lista_meteo, ignore_index=True)
    df_meteo['fecha'] = pd.to_datetime(df_meteo['fecha'])
    df_meteo = df_meteo.sort_values(by=['ubicacion_meteo', 'fecha']).set_index('fecha')

    # VENTANA DE CONTEXTO (30 DÍAS):
    # Agrupamos por estación y miramos 30 días hacia atrás para cada fecha, 
    # calculando promedios, sumas acumuladas y picos de viento para que el modelo tenga contexto ambiental.
    df_clima_30d = df_meteo.groupby('ubicacion_meteo').rolling('30D').agg({
        'temp_max': 'mean', 'temp_min': 'mean', 'precipitacion_acumulada': 'sum',
        'humedad_promedio': 'mean', 'viento_max': 'max'
    }).reset_index()

    # Renombramos las columnas para reflejar que son agregaciones mensuales
    df_clima_30d.rename(columns={'temp_max': 'temp_max_30d_prom', 'temp_min': 'temp_min_30d_prom',
                                 'precipitacion_acumulada': 'precip_acumulada_30d', 
                                 'humedad_promedio': 'humedad_30d_prom', 'viento_max': 'viento_max_30d'}, inplace=True)

    # ==========================================
    # 3. FUSIÓN FINAL DE PÍXELES CON DATOS CLIMÁTICOS
    # ==========================================
    dataset_fusionado = pd.merge(df_pixeles, df_clima_30d, on=['ubicacion_meteo', 'fecha'], how='inner')

    # Guardamos el dataset fusionado en un CSV temporal para la siguiente fase de limpieza y validación
    ruta_salida = PROCESSED_DIR / "if_temp_fusion.csv"
    dataset_fusionado.to_csv(ruta_salida, index=False)
    print(f"Dataset fusionado guardado temporalmente. Píxeles totales: {len(dataset_fusionado)}")

if __name__ == "__main__":
    fusionar_clima_if()