import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# ==========================================
# BASE_DIR calcula la ruta absoluta de la carpeta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
# PROCESSED_DIR es la carpeta donde está el dataset unificado y donde guardaremos el archivo final limpio
PROCESSED_DIR = BASE_DIR / "data" / "processed"
# REPORTS_DIR es la carpeta destinada a guardar las visualizaciones y gráficos de validación (Auditoría)
REPORTS_DIR = BASE_DIR / "reports" / "figures"

def limpiar_y_validar_datos():
    print("Iniciando Paso 3: Limpieza y Auditoría de Datos...\n")
    
    # ==========================================
    # 1. CARGA DEL DATASET UNIFICADO
    # ==========================================
    ruta_entrada = PROCESSED_DIR / "dataset_unificado.csv"
    
    # Verificamos que el archivo generado en el Paso 2 realmente exista antes de continuar
    if not ruta_entrada.exists():
        print("Error: No se encontró el archivo dataset_unificado.csv")
        return
        
    # Cargamos el archivo en un DataFrame de Pandas y contamos cuántas filas tenemos al inicio
    df = pd.read_csv(ruta_entrada)
    total_inicial = len(df)
    print(f"Registros iniciales antes de limpieza: {total_inicial}")

    # ==========================================
    # 2. FILTRADO DE NUBES E IMPUTACIÓN INTELIGENTE
    # ==========================================
    # Lista de los índices espectrales que calculamos en el Paso 1
    indices = ['NDVI', 'EVI', 'SAVI', 'NDRE', 'MTCI', 'NDMI', 'MSI', 'NBR']
    
    # 1. Eliminamos SOLAMENTE las filas donde TODOS los índices son exactamente 0 (nubes totales)
    df = df[~(df[indices] == 0).all(axis=1)]
    
    # 2. Corrección de errores de lectura del CSV: 
    # Identificamos filas donde EVI o SAVI valen 0, pero el NDVI es mayor a 0 (lo cual prueba que hay planta).
    # Esto ocurre por errores al guardar el archivo cuando faltaba la banda azul.
    mask_error_evi = (df['EVI'] == 0) & (df['NDVI'] > 0)
    mask_error_savi = (df['SAVI'] == 0) & (df['NDVI'] > 0)

    # Convertimos esos ceros engañosos de vuelta a valores nulos/vacíos (pd.NA) de Pandas
    df.loc[mask_error_evi, 'EVI'] = pd.NA
    df.loc[mask_error_savi, 'SAVI'] = pd.NA

    # 3. IMPUTACIÓN CONDICIONAL (Relleno inteligente basado en clases):
    print("Aplicando imputación inteligente para salvar registros incompletos...")
    for indice in ['EVI', 'SAVI']:
        # En lugar de borrar la fila, agrupamos los datos por la etiqueta (Ej: "1_Estres_Hidrico").
        # Si a una planta con estrés le falta el EVI, lo rellenamos con la MEDIANA del EVI de las demás plantas con estrés.
        # Esto salva el registro manteniendo la coherencia agronómica y estadística.
        df[indice] = df.groupby('etiqueta')[indice].transform(lambda x: x.fillna(x.median()))
    
    # Si quedó algún valor vacío en las parcelas "No_Etiquetadas" (que no tienen grupo), 
    # aplicamos una mediana general para que el algoritmo no falle.
    df[indices] = df[indices].fillna(df[indices].median())
    
    print(f"Registros rescatados tras imputación inteligente: {len(df)}")

    # ==========================================
    # 3. FILTRADO POR RANGOS TEÓRICOS
    # ==========================================
    # Para blindar el dataset, forzamos a que los índices respeten sus límites físicos y matemáticos.
    
    # Estos índices son divisiones simétricas, su valor nunca puede salir del rango entre -1.0 y 1.0
    indices_simetricos = ['NDVI', 'EVI', 'SAVI', 'NDRE', 'NDMI', 'NBR']
    for ind in indices_simetricos:
        # Filtramos y conservamos estrictamente las filas que estén dentro de este límite lógico
        df = df[(df[ind] >= -1.0) & (df[ind] <= 1.0)]
    
    # MTCI y MSI tienen fórmulas distintas y no pueden arrojar valores negativos en un escenario real
    indices_positivos = ['MTCI', 'MSI']
    for ind in indices_positivos:
        # Conservamos solo las filas con valores mayores o iguales a 0
        df = df[df[ind] >= 0.0]
        
    print(f"Registros tras aplicar límites matemáticos estrictos: {len(df)}")

    # ==========================================
    # 4. AUDITORÍA VISUAL CON GRÁFICOS (Solo Etiquetadas)
    # ==========================================
    # Creamos la carpeta de reportes gráficos si el sistema no la tiene creada
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filtramos el dataset para graficar SOLAMENTE las parcelas validadas por agrónomos
    # (Las "No_Etiquetadas" no nos sirven para comprobar si la matemática coincide con la realidad)
    df_etiquetado = df[df['etiqueta'] != "No_Etiquetada"].copy()
    
    if not df_etiquetado.empty:
        print("\nGenerando gráficos de validación (Boxplots) para parcelas etiquetadas...")
        
        # Configuramos el estilo visual del gráfico usando la librería Seaborn
        sns.set_theme(style="whitegrid")
        
        # Creamos una figura grande (lienzo) dividida en 2 filas y 4 columnas (para ubicar los 8 índices)
        fig, axes = plt.subplots(2, 4, figsize=(22, 12))
        fig.suptitle('Auditoría Visual de Índices Espectrales vs. Etiquetas en Campo', fontsize=18, fontweight='bold')        

        # Aplanamos la cuadrícula de gráficos para poder recorrerla fácilmente con un bucle
        axes = axes.flatten()
        
        # Dibujamos un diagrama de caja (boxplot) para evaluar la distribución de cada índice individual
        for i, indice in enumerate(indices):
            # Eje X: Las etiquetas de campo | Eje Y: El valor del índice
            sns.boxplot(x='etiqueta', y=indice, data=df_etiquetado, ax=axes[i], palette="Set2")
            axes[i].set_title(f'Distribución de {indice}', fontsize=14)
            axes[i].tick_params(axis='x', rotation=25) # Inclinamos las etiquetas para que no se superpongan
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Valor del Índice')

        # Ajustamos los márgenes automáticamente para que el gráfico quede prolijo
        plt.tight_layout()
        
        # Guardamos la imagen en alta calidad (300 dpi) lista para adjuntar en el documento del TFC
        ruta_grafico = REPORTS_DIR / "auditoria_indices.png"
        plt.savefig(ruta_grafico, dpi=300)
        print(f" -> ¡Gráfico de auditoría guardado con éxito en: {ruta_grafico}!")
    else:
        print("\nNo hay suficientes datos etiquetados para generar los gráficos.")

    # ==========================================
    # 5. GUARDADO DEL DATASET FINAL
    # ==========================================
    # Exportamos el dataset resultante, ya limpio, auditado e imputado, listo para entrenar el modelo
    ruta_salida = PROCESSED_DIR / "dataset_limpio.csv"
    df.to_csv(ruta_salida, index=False)
    print(f"\n¡Limpieza finalizada! Dataset listo para Machine Learning guardado en: {ruta_salida}")

# Bloque de ejecución principal del script
if __name__ == "__main__":
    limpiar_y_validar_datos()