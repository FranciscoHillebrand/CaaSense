import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# ==========================================
# BASE_DIR calcula la ruta absoluta de la carpeta principal de todo el sistema
BASE_DIR = Path(__file__).resolve().parent.parent
# PROCESSED_DIR es la carpeta de donde leeremos las 4 particiones de datos CSV ya limpias
PROCESSED_DIR = BASE_DIR / "data" / "processed"
# MODELS_DIR es una nueva carpeta donde guardaremos los algoritmos entrenados y las reglas matemáticas
MODELS_DIR = BASE_DIR / "models"

def preparar_datos():
    """Carga los datos, separa las variables predictoras de las etiquetas y estandariza las escalas matemáticas."""
    print("Iniciando Paso 1: Ingeniería de Características...\n")
    
    # ==========================================
    # 1. CARGA DE LAS PARTICIONES (Del Bloque 1)
    # ==========================================
    # Intentamos cargar los 4 archivos que generamos en el script anterior (data_splitting.py)
    try:
        # Conjunto de Entrenamiento (65%) para que el Random Forest aprenda
        df_train = pd.read_csv(PROCESSED_DIR / "rf_train.csv")
        # Conjunto de Validación (14%) para calibrar y ajustar el Random Forest
        df_val = pd.read_csv(PROCESSED_DIR / "rf_val.csv")
        # Conjunto de Prueba/Test (21%) (Incluye Kalena y Parcela Testigo) para el examen final
        df_test = pd.read_csv(PROCESSED_DIR / "rf_test.csv")
        # Conjunto No Etiquetado (Datos crudos) exclusivo para entrenar el Isolation Forest
        df_if = pd.read_csv(PROCESSED_DIR / "if_unlabeled.csv")
    except FileNotFoundError as e:
        # Si alguien corre este script sin haber hecho el Bloque 1, el programa avisa y se frena
        print(f"Error: Faltan archivos de datos. Ejecute la partición primero. Detalle: {e}")
        return None

    # ==========================================
    # 2. DEFINICIÓN DE FEATURES (X) Y TARGET (y)
    # ==========================================
    # Para que el modelo aprenda fisiología y clima, debemos ocultarle la "identidad" de la planta.
    # Excluimos metadatos: la fecha de la foto, el nombre de la chacra, la estación meteorológica y la respuesta final (etiqueta).
    columnas_excluir = ['fecha', 'parcela', 'ubicacion_meteo', 'etiqueta']
    
    # Automáticamente, todas las columnas que NO son metadatos se convierten en nuestras "Features" (Variables predictoras).
    # Esto agrupa a los 8 índices (NDVI, SAVI, etc.) y las 5 variables de clima (Temp, Lluvia, etc.)
    features = [col for col in df_train.columns if col not in columnas_excluir]
    print(f"Variables predictoras ({len(features)}): {features}\n")
    
    # SEPARACIÓN MATEMÁTICA: 
    # 'X' es la matriz de características (las pistas) e 'y' es el vector Target (la respuesta esperada)
    X_train, y_train = df_train[features], df_train['etiqueta']
    X_val, y_val = df_val[features], df_val['etiqueta']
    X_test, y_test = df_test[features], df_test['etiqueta']
    
    # El Isolation Forest es "No Supervisado", por lo que solo necesita las pistas (X), no tiene respuestas (y)
    X_if = df_if[features]

    # ==========================================
    # 3. ESCALADO DE DATOS (Estandarización)
    # ==========================================
    # Los algoritmos miden "distancias" matemáticas. Si la lluvia se mide en milímetros (ej. 150) 
    # y el NDVI va de -1 a 1, el algoritmo le daría 150 veces más peso a la lluvia por error.
    # StandardScaler resuelve esto: convierte todas las variables a una escala común (Media = 0, Varianza = 1).
    scaler = StandardScaler()
    
    # REGLA DE ORO CONTRA LA FUGA DE DATOS (DATA LEAKAGE): 
    # Usamos .fit_transform() SOLO en el conjunto de Entrenamiento. 
    # Esto significa: "Mirá solo los datos de Train, calculá su promedio/desviación y aplicalo".
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Para Validación, Testeo e Isolation Forest usamos SOLO .transform().
    # Esto significa: "Escalá estos datos nuevos utilizando EXACTAMENTE la misma matemática que 
    # calculaste para Entrenamiento". Así, el modelo jamás "ve" el futuro (datos de testeo) durante el cálculo.
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_if_scaled = scaler.transform(X_if)
    
    # GUARDADO DEL ESCALADOR:
    # Cuando exportemos el software (.exe) y entre una imagen satelital nueva en 2027, 
    # vamos a necesitar este exacto mismo objeto matemático para escalar la nueva imagen.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_scaler = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, ruta_scaler)
    
    print(f"-> Datos separados y escalados correctamente.")
    print(f"-> Escalador guardado en: {ruta_scaler}\n")
    
    # La función devuelve las matrices ya procesadas para inyectarlas directamente en los algoritmos
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, X_if_scaled, features

# Bloque de ejecución. Permite probar que la separación y el escalado funcionen correctamente.
if __name__ == "__main__":
    preparar_datos()