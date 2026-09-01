import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# ==========================================
# BASE_DIR calcula la ruta absoluta de la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
# PROCESSED_DIR es la carpeta donde leeremos el dataset limpio y guardaremos las particiones
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def particionar_datos():
    print("Iniciando Paso 4: Partición de Datos (Data Splitting)...\n")

    # Verificamos que el dataset limpio del Paso 3 exista
    ruta_entrada = PROCESSED_DIR / "dataset_limpio.csv"
    if not ruta_entrada.exists():
        print("Error: No se encontró el dataset_limpio.csv")
        return
        
    df = pd.read_csv(ruta_entrada)
    
    # ==========================================
    # 1. SEPARACIÓN POR TIPO DE ALGORITMO (Supervisado vs No Supervisado)
    # ==========================================
    # Isolation Forest: Algoritmo detector de anomalías ciego. 
    # Le pasamos EXCLUSIVAMENTE los datos sin etiqueta para que aprenda la "normalidad" del campo sin sesgo humano.
    df_isolation_forest = df[df['etiqueta'] == 'No_Etiquetada'].copy()
    
    # Random Forest: Algoritmo de clasificación. 
    # Le pasamos solo los registros etiquetados y validados en campo para que aprenda a distinguir clases.
    df_rf = df[df['etiqueta'] != 'No_Etiquetada'].copy()
    
    print(f"Datos para Isolation Forest (No Supervisado): {len(df_isolation_forest)} registros.")
    print(f"Datos para Random Forest (Supervisado): {len(df_rf)} registros.\n")

    # ==========================================
    # 2. DEFINICIÓN ESTRICTA DEL CONJUNTO DE PRUEBA (TEST)
    # ==========================================
    # Para evitar la "fuga de datos espacial" (Data Leakage), no separamos filas al azar para el Test,
    # sino que aislamos parcelas físicas completas que el modelo jamás verá durante el entrenamiento.
    
    # Regla 1 (Diseño del TFC): Kalena 5 y 6 reservadas obligatoriamente para evaluar transiciones de vigor.
    parcelas_test_fijo = ['Kalena_Parcela_5', 'Kalena_Parcela_6']
    
    # Regla 2 (Parcela Testigo): Buscamos dinámicamente una parcela de OTRA localidad distinta a Kalena
    # que tenga una anomalía confirmada (Plaga o Estrés). Esto probará si el modelo realmente 
    # aprendió fisiología vegetal o si solo memorizó la geografía de Kalena.
    anomalas_fuera_de_kalena = df_rf[
        (df_rf['etiqueta'].isin(['4_Plaga_Activa', '1_Estres_Hidrico'])) & 
        (~df_rf['parcela'].str.contains('Kalena'))
    ]['parcela'].unique()

    # Si encontramos parcelas que cumplan la condición, tomamos la primera y la sumamos al grupo de Test
    if len(anomalas_fuera_de_kalena) > 0:
        testigo = anomalas_fuera_de_kalena[0]
        parcelas_test_fijo.append(testigo)
        print(f"Parcela Testigo asignada automáticamente: {testigo}")
    
    # Filtramos el dataset físico: separamos las parcelas elegidas para Test y dejamos el resto
    mask_test = df_rf['parcela'].isin(parcelas_test_fijo)
    df_test_mandatorio = df_rf[mask_test].copy()
    df_resto = df_rf[~mask_test].copy()
    
    # Verificamos el volumen del conjunto de Test. El objetivo teórico del TFC es 15%.
    total_rf = len(df_rf)
    objetivo_test = int(total_rf * 0.15)

    # Si el bloque espacial fijo (Kalena 5, 6 + Testigo) supera el 15% (como ocurre en este proyecto), 
    # se prioriza la integridad física del lote y se acepta un Test mayor (~21%) antes que romper parcelas por la mitad.
    if len(df_test_mandatorio) < objetivo_test:
        # Solo en caso de no llegar al 15%, tomaríamos una muestra aleatoria para completar
        faltante = objetivo_test - len(df_test_mandatorio)
        df_resto, df_test_aleatorio = train_test_split(
            df_resto, 
            test_size=faltante, 
            stratify=df_resto['etiqueta'], # Mantiene la proporción de clases sanas/enfermas
            random_state=42
        )
        df_test = pd.concat([df_test_mandatorio, df_test_aleatorio])
    else:
        df_test = df_test_mandatorio
        
    print(f"Conjunto de Test (Prueba) consolidado: {len(df_test)} registros (Aprox 15%).")

    # ==========================================
    # 3. DIVISIÓN DE ENTRENAMIENTO Y VALIDACIÓN
    # ==========================================
    # El df_resto contiene la masa genérica de datos. De aquí separamos Validación para ajustar hiperparámetros.
    # Matemática: Para que Validación sea el 15% del total original, calculamos su peso relativo frente al 85% restante.
    proporcion_val_relativa = 0.15 / 0.85 

    # train_test_split divide aleatoriamente el resto, usando 'stratify' para asegurar que 
    # todas las etiquetas (Plaga, Estrés, Sano) estén equilibradas en ambos conjuntos.
    df_train, df_val = train_test_split(
        df_resto, 
        test_size=proporcion_val_relativa, 
        stratify=df_resto['etiqueta'], 
        random_state=42
    )
    
    print(f"Conjunto de Train (Entrenamiento): {len(df_train)} registros (Aprox 70%).")
    print(f"Conjunto de Val (Validación): {len(df_val)} registros (Aprox 15%).\n")
    
    # ==========================================
    # 4. EXPORTACIÓN BLINDADA (PREVENCIÓN DE DATA LEAKAGE)
    # ==========================================
    # Exportamos los 4 archivos finales (3 para Random Forest, 1 para Isolation Forest)
    # index=False evita que Pandas guarde la columna de numeración de filas.
    df_train.to_csv(PROCESSED_DIR / "rf_train.csv", index=False)
    df_val.to_csv(PROCESSED_DIR / "rf_val.csv", index=False)
    df_test.to_csv(PROCESSED_DIR / "rf_test.csv", index=False)
    df_isolation_forest.to_csv(PROCESSED_DIR / "if_unlabeled.csv", index=False)
    
    print("¡Éxito! Todos los conjuntos han sido exportados a data/processed/ listos para modelar.")

# Bloque de ejecución principal del script
if __name__ == "__main__":
    particionar_datos()