import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# ==========================================
# BASE_DIR calcula la ruta absoluta de la carpeta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
# PROCESSED_DIR es la carpeta donde está el CSV satelital y donde se guardará el resultado final
PROCESSED_DIR = BASE_DIR / "data" / "processed"
# METEO_DIR es la carpeta donde están guardados los archivos CSV del clima (Open-Meteo)
METEO_DIR = BASE_DIR / "data" / "meteo"

def mapear_estacion_meteorologica(parcela: str) -> str:
    """Asigna la parcela a su archivo CSV meteorológico correspondiente."""
    # Convertimos el nombre de la parcela a mayúsculas para evitar errores por minúsculas/mayúsculas
    parcela_upper = str(parcela).upper()

    # Buscamos palabras clave en el nombre de la parcela para saber qué clima le corresponde
    if "BASILIO" in parcela_upper: return "BASILIO"
    elif "CARAGUATAY" in parcela_upper: return "CARAGUATAY"
    elif "JARDIN_AMERICA" in parcela_upper: return "JARDIN_AMERICA"
    elif "KALENA" in parcela_upper: return "KALENA"
    elif "HELECHOSI_" in parcela_upper: return "LOS_HELECHOS_1"
    elif "HELECHOSII_" in parcela_upper: return "LOS_HELECHOS_2"
    elif "HELECHOSIII_" in parcela_upper: return "LOS_HELECHOS_3"
    else: return None # Si no encuentra coincidencia, devuelve nulo

def fusionar_datos():
    print("Iniciando fusión de datos espaciales y meteorológicos...\n")
    
    # ==========================================
    # 1. CARGAR EL DATASET SATELITAL
    # ==========================================
    ruta_satelital = PROCESSED_DIR / "indices_satelitales.csv"

    # Verificamos que el archivo generado en el Paso 1 realmente exista
    if not ruta_satelital.exists():
        print("Error: No se encontró el archivo indices_satelitales.csv")
        return
        
    # Leemos el archivo CSV de los índices y lo guardamos en formato tabla (DataFrame)
    df_satelital = pd.read_csv(ruta_satelital)
    # Convertimos la columna 'fecha' a un formato de tiempo oficial que Pandas pueda entender
    df_satelital['fecha'] = pd.to_datetime(df_satelital['fecha'])
    
    # Crear columna de conexión: Aplicamos la función traductora a cada fila 
    # para saber qué archivo de clima debe buscar cada parcela
    df_satelital['ubicacion_meteo'] = df_satelital['parcela'].apply(mapear_estacion_meteorologica)
    
    # ==========================================
    # 2. CARGAR Y CONSOLIDAR TODOS LOS CSV METEOROLÓGICOS
    # ==========================================
    lista_meteo = []
    # Obtenemos una lista única de las estaciones meteorológicas que realmente necesitamos usar
    estaciones = df_satelital['ubicacion_meteo'].dropna().unique()

    # Recorremos cada estación necesaria para cargar su archivo CSV
    for estacion in estaciones:
        ruta_csv_meteo = METEO_DIR / f"datos_meteorologicos_{estacion}.csv"
        if ruta_csv_meteo.exists():
            # Si el archivo existe, lo leemos
            df_temp = pd.read_csv(ruta_csv_meteo)
            # Le agregamos una columna con el nombre de la estación para poder identificar los datos luego
            df_temp['ubicacion_meteo'] = estacion
            # Guardamos esta tabla en nuestra lista temporal
            lista_meteo.append(df_temp)
        else:
            print(f"Advertencia: Falta el archivo del clima para {estacion}")
            
    # Unimos (concatenamos) todos los CSVs de clima sueltos en una sola gran tabla maestra
    df_meteo = pd.concat(lista_meteo, ignore_index=True)
    # Convertimos las fechas del clima al mismo formato de tiempo oficial
    df_meteo['fecha'] = pd.to_datetime(df_meteo['fecha'])
    
    # ==========================================
    # 3. CALCULAR LA VENTANA TEMPORAL DE 30 DÍAS
    # ==========================================
    print("Calculando ventana temporal acumulada de 30 días...")

    # Ordenamos los datos cronológicamente y por estación (requisito obligatorio para mirar hacia atrás en el tiempo)
    df_meteo = df_meteo.sort_values(by=['ubicacion_meteo', 'fecha'])
    # Transformamos la fecha en el "índice" (la columna vertebral) de la tabla para hacer cálculos temporales
    df_meteo = df_meteo.set_index('fecha')
    
    # Agrupamos por estación y usamos rolling('30D') para mirar 30 días hacia atrás.
    # .agg() decide qué operación matemática hacer con cada variable climática de ese mes
    df_clima_30d = df_meteo.groupby('ubicacion_meteo').rolling('30D').agg({
        'temp_max': 'mean',                  # Promedia la temperatura máxima de los 30 días
        'temp_min': 'mean',                  # Promedia la temperatura mínima de los 30 días
        'precipitacion_acumulada': 'sum',    # Suma toda el agua que cayó en esos 30 días
        'humedad_promedio': 'mean',          # Promedia la humedad del mes
        'viento_max': 'max'                  # Extrae la ráfaga de viento más fuerte de todo el mes
    }).reset_index() # Volvemos a convertir la fecha y la ubicación en columnas normales
    
    # Renombrar columnas para dejar en claro que ya no son datos diarios, sino mensuales (30 días)
    df_clima_30d.rename(columns={
        'temp_max': 'temp_max_30d_prom',
        'temp_min': 'temp_min_30d_prom',
        'precipitacion_acumulada': 'precip_acumulada_30d',
        'humedad_promedio': 'humedad_30d_prom',
        'viento_max': 'viento_max_30d'
    }, inplace=True)
    
    # Redondear todas las columnas numéricas a 2 decimales para mantener el dataset limpio
    cols_numericas = df_clima_30d.columns.drop(['ubicacion_meteo', 'fecha'])
    df_clima_30d[cols_numericas] = df_clima_30d[cols_numericas].round(2)
    
    # ==========================================
    # 4. SINCRONIZACIÓN FINAL (JOIN)
    # ==========================================
    print("Sincronizando registros por fecha y ubicación...")

    # pd.merge cruza ambas tablas. 
    # 'on' indica que la llave para unirlas es que coincidan exactamente en fecha y ubicación.
    # 'how=left' significa que mantenemos intacta la tabla satelital y le pegamos el clima al lado.
    dataset_final = pd.merge(
        df_satelital, 
        df_clima_30d, 
        on=['ubicacion_meteo', 'fecha'], 
        how='left'
    )
    
    # Guardar el resultado
    ruta_salida = PROCESSED_DIR / "dataset_unificado.csv"
    dataset_final.to_csv(ruta_salida, index=False)

    print(f"\n¡Éxito! Dataset unificado guardado en: {ruta_salida}")
    print(f"Total de registros: {len(dataset_final)}")

# Este bloque asegura que el código solo se ejecute si lanzamos este script directamente
if __name__ == "__main__":
    fusionar_datos()