import pandas as pd
from pathlib import Path

# Configuramos la ruta base para que encuentre la carpeta reports
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

print("\n==================================================")
print(" CALCULANDO PROMEDIOS DE LAS 5 ITERACIONES...")
print("==================================================")

dataframes_metricas = []

for i in range(1, 6):
    ruta_csv = REPORTS_DIR / "figures" / f"iteracion_{i}" / "metricas_rf.csv"
    
    if ruta_csv.exists():
        df = pd.read_csv(ruta_csv, index_col=0)
        dataframes_metricas.append(df)
    else:
        print(f"Falta el archivo: {ruta_csv}")
        
if dataframes_metricas:
    df_promedio = sum(dataframes_metricas) / len(dataframes_metricas)
    df_promedio = df_promedio.round(4)
    
    ruta_promedio = REPORTS_DIR / "figures" / "promedio_final_rf.csv"
    df_promedio.to_csv(ruta_promedio)
    
    print("\nPROMEDIOS CALCULADOS CON EXITO!")
    print(f"-> Archivo guardado en: {ruta_promedio}")
    print("\nVista previa del rendimiento promedio:")
    print(df_promedio[['precision', 'recall', 'f1-score']])
else:
    print("No se encontraron los archivos CSV para promediar.")