import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS (Basado en tu script original)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def generar_datos_sinteticos():
    print("Iniciando partición espacial y generación sintética (SMOTE)...\n")

    # 1. Cargar el dataset
    df = pd.read_csv(PROCESSED_DIR / "dataset_limpio.csv")
    
    # Filtramos No_Etiquetada como en tu script original
    df_rf = df[df['etiqueta'] != 'No_Etiquetada'].copy()

    # ==========================================
    # 2. DEFINICIÓN ESTRICTA DEL TEST (Cero Leakage)
    # ==========================================
    parcelas_test_fijo = ['Kalena_Parcela_5', 'Kalena_Parcela_6']
    
    # Obtenemos todas las parcelas que tienen Plaga Activa (excluyendo Kalena)
    parcelas_plaga = df_rf[(df_rf['etiqueta'] == '4_Plaga_Activa') & (~df_rf['parcela'].isin(parcelas_test_fijo))]['parcela'].unique()
    
    # Regla 3: EXACTAMENTE 4 parcelas de Plaga Activa para Test
    np.random.seed(42)
    test_plaga = np.random.choice(parcelas_plaga, 4, replace=False)
    
    # Consolidamos las parcelas de Test
    test_parcelas = list(parcelas_test_fijo) + list(test_plaga)

    # ==========================================
    # 3. DIVISIÓN ESPACIAL PARA TRAIN Y VAL (70/15)
    # ==========================================
    # Repartimos el RESTO de las parcelas de plaga
    resto_parcelas_plaga = np.setdiff1d(parcelas_plaga, test_plaga)
    train_plaga, val_plaga = train_test_split(resto_parcelas_plaga, test_size=0.15, random_state=42)
    
    # Repartimos todas las demás parcelas (Sano, Estrés, etc.)
    parcelas_otras = df_rf[(df_rf['etiqueta'] != '4_Plaga_Activa') & (~df_rf['parcela'].isin(test_parcelas))]['parcela'].unique()
    train_otras, val_otras = train_test_split(parcelas_otras, test_size=0.15, random_state=42)
    
    # Ensamblamos
    train_parcelas = np.concatenate([train_plaga, train_otras])
    val_parcelas = np.concatenate([val_plaga, val_otras])
    
    df_train = df_rf[df_rf['parcela'].isin(train_parcelas)].copy()
    df_val = df_rf[df_rf['parcela'].isin(val_parcelas)].copy()
    df_test = df_rf[df_rf['parcela'].isin(test_parcelas)].copy()

    print(f"--- Volumen REAL de registros ---")
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # ==========================================
    # 4. APLICACIÓN DE SMOTE SÓLO AL ENTRENAMIENTO
    # ==========================================
    # Separar columnas numéricas (para la matemática) y de texto (para preservar metadatos)
    numeric_cols = df_train.select_dtypes(include=['number']).columns.tolist()
    text_cols = df_train.select_dtypes(exclude=['number']).columns.tolist()
    if 'etiqueta' in text_cols: 
        text_cols.remove('etiqueta')

    X_train_num = df_train[numeric_cols]
    y_train = df_train['etiqueta']

    print(f"\nDistribución REAL en Train:\n{y_train.value_counts()}")

    # Aplicamos SMOTE clásico SOLO para la plaga.
    conteo_real = y_train.value_counts().to_dict()
    
    estrategia = {
        '3_Baja_Densidad_Foliar': conteo_real['3_Baja_Densidad_Foliar'],
        '0_Sano_Normal': conteo_real['0_Sano_Normal'],
        '2_Deficiencia_Nutricional': conteo_real['2_Deficiencia_Nutricional'],
        '1_Estres_Hidrico': conteo_real['1_Estres_Hidrico'],
        
        # Le sumamos los 20 registros sintéticos (para compensar las 4 parcelas reales que mandaste al test)
        '4_Plaga_Activa': conteo_real['4_Plaga_Activa'] + 20 
    }

    smote = SMOTE(sampling_strategy=estrategia, random_state=42)
    X_res_num, y_res = smote.fit_resample(X_train_num, y_train)

    # Identificamos EXCLUSIVAMENTE las filas nuevas generadas
    num_reales = len(X_train_num)
    X_new_num = X_res_num.iloc[num_reales:].copy()
    y_new = y_res.iloc[num_reales:].copy()

    # Reconstruimos las filas sintéticas con TODAS las columnas
    df_new = pd.DataFrame(X_new_num, columns=numeric_cols)
    df_new['etiqueta'] = y_new
    for col in text_cols:
        df_new[col] = 'Dato_Sintetico' # Rellenamos los metadatos de texto

    # Unimos los datos 100% reales intactos con los sintéticos nuevos
    df_train_final = pd.concat([df_train, df_new], ignore_index=True)
    
    # Aseguramos el orden original de las columnas
    df_train_final = df_train_final[df_rf.columns]

    print(f"\nDistribución SINTÉTICA/BALANCEADA en Train:\n{df_train_final['etiqueta'].value_counts()}")

    # ==========================================
    # 5. EXPORTACIÓN
    # ==========================================
    df_train_final.to_csv(PROCESSED_DIR / "indices_satelitales_sinteticos.csv", index=False)
    df_val.to_csv(PROCESSED_DIR / "val_real.csv", index=False)
    df_test.to_csv(PROCESSED_DIR / "test_real.csv", index=False)

    print("\n Pipeline completado. Los conjuntos conservan todas las columnas y no hay data leakage.")

if __name__ == "__main__":
    generar_datos_sinteticos()