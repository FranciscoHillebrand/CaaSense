import os
import time
import requests
import pandas as pd
from pathlib import Path

# Calculamos automáticamente la ruta raíz del proyecto de forma absoluta
# __file__ es este script (data_ingestion.py). .parent.parent sube a la carpeta raíz (TFC)
BASE_DIR = Path(__file__).resolve().parent.parent 
DEFAULT_OUTPUT = BASE_DIR / "data" / "meteo"

def descargar_datos_meteorologicos(ubicaciones: dict, fecha_inicio: str, fecha_fin: str, output_dir: Path = DEFAULT_OUTPUT):
    """
    Descarga datos meteorológicos históricos de Open-Meteo y los guarda como CSV.
    """
    
    # 1. Asegurarnos de que la carpeta de destino exista
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    print(f"Iniciando descarga de datos meteorológicos en '{output_dir}'...\n")
    
    # 2. Bucle de descarga y procesamiento
    for nombre, (lat, lon) in ubicaciones.items():
        print(f"[{nombre}] Solicitando datos (Lat: {lat}, Lon: {lon})...")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": fecha_inicio,
            "end_date": fecha_fin,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "America/Argentina/Buenos_Aires"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status() 
            
            datos_json = response.json()
            hourly_data = datos_json["hourly"]
            
            df = pd.DataFrame({
                "fecha_hora": pd.to_datetime(hourly_data["time"]),
                "temperatura": hourly_data["temperature_2m"],
                "humedad": hourly_data["relative_humidity_2m"],
                "precipitacion": hourly_data["precipitation"],
                "viento": hourly_data["wind_speed_10m"]
            })
            
            df.set_index("fecha_hora", inplace=True)
            
            df_diario = df.resample("D").agg(
                temp_max=("temperatura", "max"),
                temp_min=("temperatura", "min"),
                precipitacion_acumulada=("precipitacion", "sum"),
                humedad_promedio=("humedad", "mean"),
                viento_max=("viento", "max") 
            ).reset_index()
            
            df_diario.rename(columns={"fecha_hora": "fecha"}, inplace=True)
            
            # SOLUCIÓN AL WARNING: Redondeamos solo las columnas numéricas
            cols_numericas = ["temp_max", "temp_min", "precipitacion_acumulada", "humedad_promedio", "viento_max"]
            df_diario[cols_numericas] = df_diario[cols_numericas].round(2)
            
            # 3. Guardado en la ruta absoluta
            ruta_archivo = output_dir / f"datos_meteorologicos_{nombre}.csv"
            df_diario.to_csv(ruta_archivo, index=False, encoding='utf-8')
            
            print(f" -> ¡Éxito! Guardado en: {ruta_archivo} ({len(df_diario)} días procesados)\n")
            
        except requests.exceptions.RequestException as e:
            print(f" -> ERROR al descargar {nombre}: Hubo un problema de conexión - {e}\n")
            
        time.sleep(1)

    print("¡Proceso general finalizado!")


# ==========================================
# BLOQUE DE PRUEBA
# ==========================================
if __name__ == "__main__":
    ubicaciones_prueba = {
        "KALENA": (-27.985814, -55.606944),
        "BASILIO": (-27.387333, -55.066577),
        "LOS_HELECHOS_1": (-27.597801, -55.071163),
        "LOS_HELECHOS_2": (-27.598970, -55.073636),
        "LOS_HELECHOS_3": (-27.573340, -55.062446),
        "JARDIN_AMERICA": (-26.996706, -55.265297),
        "CARAGUATAY": (-26.696611, -54.731917)
    }
    
    descargar_datos_meteorologicos(
        ubicaciones=ubicaciones_prueba,
        fecha_inicio="2018-12-01",
        fecha_fin="2026-06-01"
    )