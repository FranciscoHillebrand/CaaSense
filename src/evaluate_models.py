import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "figures"
# 1. Crear carpeta exclusiva para cada iteración de validación de las parcelas testigo, para mantener un historial de resultados.
iteracion_actual = os.environ.get("ITERACION_CV", "1")
ITER_DIR = REPORTS_DIR / f"iteracion_{iteracion_actual}"
ITER_DIR.mkdir(parents=True, exist_ok=True)

def evaluar_arquitectura_dual():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar el motor matemático (Escalador) y los modelos
    try:
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        rf_model = joblib.load(MODELS_DIR / "rf_optimizado.pkl")
        if_model = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    except FileNotFoundError:
        print("Error: No se encontraron los modelos entrenados en la carpeta models/.")
        return

    # Columnas a excluir del escalado en ambos datasets
    columnas_excluir = ['fecha', 'parcela', 'ubicacion_meteo', 'etiqueta']

    print("==========================================================")
    print(" PARTE 1: EVALUACIÓN SUPERVISADA (ESCALA PARCELA - RF)")
    print("==========================================================\n")
    
    # Cargar conjunto de Test de Random Forest (Medianas)
    df_rf_test = pd.read_csv(PROCESSED_DIR / "rf_test.csv")
    features_rf = [c for c in df_rf_test.columns if c not in columnas_excluir]
    
    X_rf_test = df_rf_test[features_rf]
    y_rf_test = df_rf_test['etiqueta']
    
    # Escalar y Predecir
    X_rf_scaled = scaler.transform(X_rf_test)
    y_rf_pred = rf_model.predict(X_rf_scaled)
    
    # Reporte de Clasificación
    print("--- Reporte de Desempeño Final (Kalena 5, 6 y Testigo) ---")
    # Guardar Reporte del Random Forest
    reporte_rf = classification_report(y_rf_test, y_rf_pred)
    print("--- Reporte RF ---")
    print(reporte_rf)
    with open(ITER_DIR / "reporte_rf_completo.txt", "w") as f:
        f.write("Reporte de Clasificación - Random Forest\n")
        f.write("========================================\n")
        f.write(reporte_rf)

    # Guardar métricas estructuradas para promediar después
    reporte_dict = classification_report(y_rf_test, y_rf_pred, output_dict=True)
    df_metricas = pd.DataFrame(reporte_dict).transpose()
    df_metricas.to_csv(ITER_DIR / "metricas_rf.csv")
    
    # Generar Matriz de Confusión Visual
    cm = confusion_matrix(y_rf_test, y_rf_pred)
    clases = sorted(y_rf_test.unique())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.title('Matriz de Confusión - Random Forest (Optimizado)', fontsize=16, pad=20)
    plt.ylabel('Diagnóstico Real (Agrónomo)', fontsize=12)
    plt.xlabel('Predicción del Modelo (IA)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    ruta_matriz = ITER_DIR / "matriz_confusion_final.png"
    plt.savefig(ruta_matriz, dpi=300)
    print(f"-> Matriz de confusión exportada con éxito a: {ruta_matriz}\n")

    print("==========================================================")
    print(" PARTE 2: AUDITORÍA NO SUPERVISADA (ESCALA PÍXEL - IF)")
    print("==========================================================\n")
    
    # Cargar conjunto de Test de Isolation Forest (Píxeles puros de las mismas parcelas)
    df_if_test = pd.read_csv(PROCESSED_DIR / "if_test.csv")
    
    # Asegurar el mismo orden de variables predictoras
    X_if_test = df_if_test[features_rf]
    
    # Escalar y Predecir (-1 = Anomalía, 1 = Normal)
    X_if_scaled = scaler.transform(X_if_test)
    if_predicciones = if_model.predict(X_if_scaled)
    
    # Integrar predicciones al dataframe para análisis
    df_if_test['if_pred'] = if_predicciones
    # Mapear a texto para mayor claridad
    df_if_test['estado_pixel'] = np.where(df_if_test['if_pred'] == -1, 'Anómalo', 'Normal')
    
    # Agrupar resultados para demostrar la variabilidad intra-parcela
    resumen_if = df_if_test.groupby(['parcela', 'etiqueta', 'estado_pixel']).size().unstack(fill_value=0)
    
    # Calcular el porcentaje real de afectación
    if 'Anómalo' not in resumen_if.columns:
        resumen_if['Anómalo'] = 0
    if 'Normal' not in resumen_if.columns:
        resumen_if['Normal'] = 0
        
    resumen_if['Total_Pixeles'] = resumen_if['Normal'] + resumen_if['Anómalo']
    resumen_if['%_Afectado'] = (resumen_if['Anómalo'] / resumen_if['Total_Pixeles'] * 100).round(2)
    
    # Mostrar resultados agronómicos
    print("--- Variabilidad Intra-Parcela detectada por Isolation Forest ---")
    # Abrimos el archivo de texto y vamos escribiendo los resultados ahí y en consola
    with open(ITER_DIR / "auditoria_isolation_forest.txt", "w", encoding="utf-8") as f_if:
        f_if.write("Resultados a nivel píxel (10x10m) - Isolation Forest\n")
        f_if.write("====================================================\n")
        
        for index, row in resumen_if.iterrows():
            parcela, diagnostico_oficial = index
            
            # Preparamos el texto
            texto_parcela = (
                f"\nParcela: {parcela}\n"
                f"Diagnóstico general del Agrónomo: {diagnostico_oficial}\n"
                f"Auditoría a nivel 10x10m: {row['Total_Pixeles']} píxeles escaneados.\n"
                f"-> Píxeles Sanos detectados: {row['Normal']}\n"
                f"-> Píxeles Anómalos detectados: {row['Anómalo']} ({row['%_Afectado']}%)\n"
            )
            
            # Validación lógica
            if "Sano" in diagnostico_oficial and row['%_Afectado'] < 20:
                texto_parcela += "   [ÉXITO] El IF confirma que el lote está mayormente sano.\n"
            elif "Sano" not in diagnostico_oficial and row['%_Afectado'] >= 15:
                texto_parcela += "   [ÉXITO] El IF localizó la anomalía diagnosticada espacialmente.\n"
            else:
                texto_parcela += "   [OBSERVACIÓN] Revisar umbrales o falsos positivos en esta parcela.\n"
            
            # Imprimimos en consola para que lo veas correr, y lo guardamos en el txt
            print(texto_parcela)
            f_if.write(texto_parcela)

if __name__ == "__main__":
    evaluar_arquitectura_dual()