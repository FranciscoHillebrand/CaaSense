import os
import time
import requests
import pandas as pd
from pathlib import Path

# Calculamos automáticamente la ruta raíz del proyecto de forma absoluta
# __file__ es este script (data_ingestion.py). .parent.parent sube a la carpeta raíz (TFC)
BASE_DIR = Path(__file__).resolve().parent.parent 
# Carpeta por defecto donde se guardarán los CSV descargados: <raiz_proyecto>/data/meteo
DEFAULT_OUTPUT = BASE_DIR / "data" / "meteo"

def descargar_datos_meteorologicos(ubicaciones: dict, fecha_inicio: str, fecha_fin: str, output_dir: Path = DEFAULT_OUTPUT):
    """
    Descarga datos meteorológicos históricos de Open-Meteo y los guarda como CSV.
 
    Parámetros:
        ubicaciones (dict): diccionario con formato {"NOMBRE": (latitud, longitud), ...}
                             indicando cada punto geográfico a consultar.
        fecha_inicio (str): fecha de inicio del rango a descargar, en formato "YYYY-MM-DD".
        fecha_fin (str): fecha de fin del rango a descargar, en formato "YYYY-MM-DD".
        output_dir (Path): carpeta donde se guardarán los archivos CSV resultantes
                            (por defecto, DEFAULT_OUTPUT).
    """
    
    # 1. Asegurarnos de que la carpeta de destino exista
    # parents=True crea también las carpetas intermedias si no existen
    # exist_ok=True evita que falle si la carpeta ya existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # URL base de la API histórica de Open-Meteo
    url = "https://archive-api.open-meteo.com/v1/archive"
    print(f"Iniciando descarga de datos meteorológicos en '{output_dir}'...\n")
    
    # 2. Bucle de descarga y procesamiento
    # Recorremos cada ubicación del diccionario, descargando y procesando sus datos por separado
    for nombre, (lat, lon) in ubicaciones.items():
        print(f"[{nombre}] Solicitando datos (Lat: {lat}, Lon: {lon})...")

        # Parámetros que se envían a la API de Open-Meteo:
        # - latitude/longitude: coordenadas del punto a consultar
        # - start_date/end_date: rango de fechas a descargar
        # - hourly: variables horarias solicitadas (temperatura, humedad, precipitación, viento)
        # - timezone: zona horaria en la que se devuelven las marcas de tiempo
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": fecha_inicio,
            "end_date": fecha_fin,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "America/Argentina/Buenos_Aires"
        }
        
        try:
            # Hacemos la petición GET a la API con los parámetros definidos arriba
            response = requests.get(url, params=params)
            # Si la respuesta HTTP indica error (4xx o 5xx), lanza una excepción
            response.raise_for_status() 

            # Convertimos la respuesta a JSON y extraemos la sección de datos horarios
            datos_json = response.json()
            hourly_data = datos_json["hourly"]

            # Armamos un DataFrame con los datos horarios devueltos por la API
            df = pd.DataFrame({
                "fecha_hora": pd.to_datetime(hourly_data["time"]),
                "temperatura": hourly_data["temperature_2m"],
                "humedad": hourly_data["relative_humidity_2m"],
                "precipitacion": hourly_data["precipitation"],
                "viento": hourly_data["wind_speed_10m"]
            })

            # Usamos la fecha_hora como índice para poder agrupar por día
            df.set_index("fecha_hora", inplace=True)

            # Agrupamos los datos horarios en datos diarios, aplicando una agregación distinta
            # a cada variable:
            # - temp_max / temp_min: valores máximo y mínimo de temperatura del día
            # - precipitacion_acumulada: suma de la precipitación horaria (total del día)
            # - humedad_promedio: promedio de humedad relativa del día
            # - viento_max: velocidad máxima de viento registrada en el día
            df_diario = df.resample("D").agg(
                temp_max=("temperatura", "max"),
                temp_min=("temperatura", "min"),
                precipitacion_acumulada=("precipitacion", "sum"),
                humedad_promedio=("humedad", "mean"),
                viento_max=("viento", "max") 
            ).reset_index()

            # Renombramos la columna del índice reseteado ("fecha_hora") a "fecha",
            # ya que ahora representa un día completo y no una fecha con hora
            df_diario.rename(columns={"fecha_hora": "fecha"}, inplace=True)
            
            # SOLUCIÓN A UN POSIBLE WARNING: Redondeamos solo las columnas numéricas
            # (evita intentar redondear la columna "fecha", que no es numérica)
            cols_numericas = ["temp_max", "temp_min", "precipitacion_acumulada", "humedad_promedio", "viento_max"]
            df_diario[cols_numericas] = df_diario[cols_numericas].round(2)
            
            # 3. Guardado en la ruta absoluta
            # Cada ubicación se guarda en su propio archivo CSV, nombrado según la ubicación
            ruta_archivo = output_dir / f"datos_meteorologicos_{nombre}.csv"
            df_diario.to_csv(ruta_archivo, index=False, encoding='utf-8')
            
            print(f" -> ¡Éxito! Guardado en: {ruta_archivo} ({len(df_diario)} días procesados)\n")
            
        except requests.exceptions.RequestException as e:
            # Capturamos cualquier error relacionado con la petición HTTP
            # (problemas de conexión, timeouts, códigos de error, etc.)
            # y continuamos con la siguiente ubicación en lugar de detener todo el proceso
            print(f" -> ERROR al descargar {nombre}: Hubo un problema de conexión - {e}\n")

        # Pequeña pausa entre peticiones para no saturar la API con solicitudes muy seguidas    
        time.sleep(1)

    print("¡Proceso general finalizado!")


# ==========================================
# BLOQUE DE PRUEBA
# ==========================================
# Este bloque solo se ejecuta si el script se corre directamente
# (no se ejecuta si el archivo es importado como módulo desde otro script)
if __name__ == "__main__":
    # Diccionario de ubicaciones de prueba: nombre -> (latitud, longitud)
    ubicaciones_prueba = {
        "KALENA": (-27.985814, -55.606944),
        "BASILIO": (-27.387333, -55.066577),
        "LOS_HELECHOS_1": (-27.597801, -55.071163),
        "LOS_HELECHOS_2": (-27.598970, -55.073636),
        "LOS_HELECHOS_3": (-27.573340, -55.062446),
        "JARDIN_AMERICA": (-26.996706, -55.265297),
        "CARAGUATAY": (-26.696611, -54.731917)
    }

    # Llamamos a la función principal para descargar los datos históricos
    # de todas las ubicaciones definidas, entre las fechas indicadas
    descargar_datos_meteorologicos(
        ubicaciones=ubicaciones_prueba,
        fecha_inicio="2018-12-01",
        fecha_fin="2026-06-01"
    )