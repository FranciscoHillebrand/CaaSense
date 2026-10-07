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
# CONFIGURACIÓN DE RUTAS (VERSIÓN SMOTE)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
# APUNTAMOS A LOS MODELOS SMOTE
MODELS_DIR = BASE_DIR / "models_smote"
# NUEVA CARPETA para los reportes de SMOTE
REPORTS_DIR = BASE_DIR / "reports" / "figures_smote"

iteracion_actual = os.environ.get("ITERACION_CV", "1")
ITER_DIR = REPORTS_DIR / f"iteracion_{iteracion_actual}"
ITER_DIR.mkdir(parents=True, exist_ok=True)

def evaluar_arquitectura_dual():
    try:
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        rf_model = joblib.load(MODELS_DIR / "rf_optimizado.pkl")
        if_model = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    except FileNotFoundError:
        print("Error: No se encontraron los modelos entrenados en models_smote/.")
        return

    columnas_excluir = ['fecha', 'parcela', 'ubicacion_meteo', 'etiqueta']

    print("==========================================================")
    print(" PARTE 1: EVALUACIÓN SUPERVISADA (ESCALA PARCELA - RF)")
    print("==========================================================\n")
    
    # LECTURA DEL TEST REAL (El que tiene las 4 parcelas de plaga intactas)
    df_rf_test = pd.read_csv(PROCESSED_DIR / "test_real.csv")
    features_rf = [c for c in df_rf_test.columns if c not in columnas_excluir]
    
    X_rf_test = df_rf_test[features_rf]
    y_rf_test = df_rf_test['etiqueta']
    
    X_rf_scaled = scaler.transform(X_rf_test)
    y_rf_pred = rf_model.predict(X_rf_scaled)
    
    reporte_rf = classification_report(y_rf_test, y_rf_pred)
    print("--- Reporte RF (SMOTE) ---")
    print(reporte_rf)
    with open(ITER_DIR / "reporte_rf_completo.txt", "w") as f:
        f.write("Reporte de Clasificación - Random Forest (SMOTE)\n")
        f.write("================================================\n")
        f.write(reporte_rf)

    reporte_dict = classification_report(y_rf_test, y_rf_pred, output_dict=True)
    df_metricas = pd.DataFrame(reporte_dict).transpose()
    df_metricas.to_csv(ITER_DIR / "metricas_rf.csv")
    
    cm = confusion_matrix(y_rf_test, y_rf_pred)
    clases = sorted(y_rf_test.unique())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.title('Matriz de Confusión - RF Optimizado (SMOTE)', fontsize=16, pad=20)
    plt.ylabel('Diagnóstico Real (Agrónomo)', fontsize=12)
    plt.xlabel('Predicción del Modelo (IA)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    ruta_matriz = ITER_DIR / "matriz_confusion_final.png"
    plt.savefig(ruta_matriz, dpi=300)
    print(f"-> Matriz exportada con éxito a: {ruta_matriz}\n")

    print("==========================================================")
    print(" PARTE 2: AUDITORÍA NO SUPERVISADA (ESCALA PÍXEL - IF)")
    print("==========================================================\n")
    
    df_if_test = pd.read_csv(PROCESSED_DIR / "if_test_sintetico.csv")
    X_if_test = df_if_test[features_rf]
    
    X_if_scaled = scaler.transform(X_if_test)
    if_predicciones = if_model.predict(X_if_scaled)
    
    df_if_test['if_pred'] = if_predicciones
    df_if_test['estado_pixel'] = np.where(df_if_test['if_pred'] == -1, 'Anómalo', 'Normal')
    
    resumen_if = df_if_test.groupby(['parcela', 'etiqueta', 'estado_pixel']).size().unstack(fill_value=0)
    
    if 'Anómalo' not in resumen_if.columns:
        resumen_if['Anómalo'] = 0
    if 'Normal' not in resumen_if.columns:
        resumen_if['Normal'] = 0
        
    resumen_if['Total_Pixeles'] = resumen_if['Normal'] + resumen_if['Anómalo']
    resumen_if['%_Afectado'] = (resumen_if['Anómalo'] / resumen_if['Total_Pixeles'] * 100).round(2)
    
    print("--- Variabilidad Intra-Parcela detectada por Isolation Forest ---")
    with open(ITER_DIR / "auditoria_isolation_forest.txt", "w", encoding="utf-8") as f_if:
        f_if.write("Resultados a nivel píxel (10x10m) - Isolation Forest\n")
        f_if.write("====================================================\n")
        for index, row in resumen_if.iterrows():
            parcela, diagnostico_oficial = index
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

            print(texto_parcela)
            f_if.write(texto_parcela)

if __name__ == "__main__":
    evaluar_arquitectura_dual()