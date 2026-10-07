import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS (VERSIÓN SMOTE)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
# NUEVA CARPETA para no pisar tus modelos reales
MODELS_DIR = BASE_DIR / "models_smote"

def preparar_datos():
    print("Iniciando Paso 1: Ingeniería de Características (Versión SMOTE)...\n")
    
    try:
        # LECTURA DE LOS NUEVOS CSV SINTÉTICOS Y REALES
        df_train = pd.read_csv(PROCESSED_DIR / "indices_satelitales_sinteticos.csv")
        df_val = pd.read_csv(PROCESSED_DIR / "val_real.csv")
        df_test = pd.read_csv(PROCESSED_DIR / "test_real.csv")
        # El Isolation Forest usa los píxeles crudos de siempre porque no lleva datos sintéticos
        df_if = pd.read_csv(PROCESSED_DIR / "if_pixels_unlabeled_sintetico.csv")
    except FileNotFoundError as e:
        print(f"Error: Faltan archivos de datos. Detalle: {e}")
        return None

    columnas_excluir = ['fecha', 'parcela', 'ubicacion_meteo', 'etiqueta']
    features = [col for col in df_train.columns if col not in columnas_excluir]
    print(f"Variables predictoras ({len(features)}): {features}\n")
    
    X_train, y_train = df_train[features], df_train['etiqueta']
    X_val, y_val = df_val[features], df_val['etiqueta']
    X_test, y_test = df_test[features], df_test['etiqueta']
    X_if = df_if[features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_if_scaled = scaler.transform(X_if)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_scaler = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, ruta_scaler)
    
    print(f"-> Datos separados y escalados correctamente.")
    print(f"-> Escalador guardado en: {ruta_scaler}\n")
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, X_if_scaled, features

def entrenar_isolation_forest(X_if_scaled):
    print("Iniciando Paso 2: Entrenamiento de Isolation Forest (No Supervisado)...\n")
    modelo_if = IsolationForest(n_estimators=100, contamination=0.10, random_state=42, n_jobs=-1)
    modelo_if.fit(X_if_scaled)
    
    ruta_modelo = MODELS_DIR / "isolation_forest.pkl"
    joblib.dump(modelo_if, ruta_modelo)
    print(f"-> ¡Detector de Anomalías entrenado y guardado en: {ruta_modelo}!\n")
    return modelo_if

def entrenar_random_forest_base(X_train_scaled, y_train, X_val_scaled, y_val):
    print("Iniciando Paso 3: Entrenamiento Base del Random Forest Classifier...\n")
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_base.fit(X_train_scaled, y_train)
    
    y_pred_val = rf_base.predict(X_val_scaled)
    print("\n--- Reporte de Clasificación Base (Conjunto de Validación REAL) ---")
    print(classification_report(y_val, y_pred_val))
    
    ruta_modelo_base = MODELS_DIR / "rf_base.pkl"
    joblib.dump(rf_base, ruta_modelo_base)
    return rf_base

def optimizar_random_forest(X_train_scaled, y_train, X_val_scaled, y_val):
    print("Iniciando Paso 4: Optimización de Hiperparámetros (Grid Search)...\n")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_template = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf_template, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro', verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    mejor_rf = grid_search.best_estimator_
    print(f"\nLos hiperparámetros óptimos encontrados son:\n{grid_search.best_params_}\n")
    
    y_pred_optimizado = mejor_rf.predict(X_val_scaled)
    print("\n--- Reporte de Clasificación OPTIMIZADO (Conjunto de Validación REAL) ---")
    print(classification_report(y_val, y_pred_optimizado))
    
    ruta_modelo_optimizado = MODELS_DIR / "rf_optimizado.pkl"
    joblib.dump(mejor_rf, ruta_modelo_optimizado)
    return mejor_rf

if __name__ == "__main__":
    datos = preparar_datos()
    if datos is not None:
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, X_if_scaled, features = datos
        modelo_anomalias = entrenar_isolation_forest(X_if_scaled)
        modelo_rf_base = entrenar_random_forest_base(X_train_scaled, y_train, X_val_scaled, y_val)
        modelo_rf_optimizado = optimizar_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)