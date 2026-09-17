import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# ==========================================
# Resolvemos la ruta absoluta para garantizar que el script funcione en cualquier computadora sin fallar
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def limpiar_pixeles_if():
    print("IF Fase 3: Limpieza Final y Aplicación de Rangos Teóricos...\n")

    # ==========================================
    # 1. CARGA DEL DATASET FUSIONADO DE PÍXELES
    # ==========================================
    ruta_entrada = PROCESSED_DIR / "if_temp_fusion.csv"
    dataset_final = pd.read_csv(ruta_entrada)

    # Blindaje estadístico: El Isolation Forest falla si se le pasa un valor 'NaN'.
    # Limpiamos cualquier fila que haya quedado con datos incompletos.
    dataset_final = dataset_final.dropna()

    # ==========================================
    # 3. FILTRADO POR RANGOS TEÓRICOS
    # ==========================================
    # Para blindar el dataset, forzamos a que los índices respeten sus límites físicos y matemáticos.
        
    # Estos índices son divisiones simétricas, su valor nunca puede salir del rango entre -1.0 y 1.0
    indices_simetricos = ['NDVI', 'EVI', 'SAVI', 'NDRE', 'NDMI', 'NBR']
    for ind in indices_simetricos:
        dataset_final = dataset_final[(dataset_final[ind] >= -1.0) & (dataset_final[ind] <= 1.0)]

    # MTCI y MSI tienen fórmulas distintas y no pueden arrojar valores negativos en un escenario real
    indices_positivos = ['MTCI', 'MSI']
    for ind in indices_positivos:
        # Conservamos solo las filas con valores mayores o iguales a 0
        dataset_final = dataset_final[dataset_final[ind] >= 0.0]

    # Guardamos el dataset final limpio y listo para ser usado por el Isolation Forest
    ruta_salida = PROCESSED_DIR / "if_pixels_unlabeled.csv"
    dataset_final.to_csv(ruta_salida, index=False)
    print(f"\n¡Listo! Píxeles limpios listos para Isolation Forest: {len(dataset_final)}")
    

if __name__ == "__main__":
    limpiar_pixeles_if()