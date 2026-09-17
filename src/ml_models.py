import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
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
    # 1. CARGA DE LAS PARTICIONES 
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
        df_if = pd.read_csv(PROCESSED_DIR / "if_pixels_unlabeled.csv")
    except FileNotFoundError as e:
        # Si alguien corre este script sin haber hecho el Bloque 1, el programa avisa y se frena
        print(f"Error: Faltan archivos de datos. Ejecute la partición primero. Detalle: {e}")
        return None

    # ==========================================
    # 2. DEFINICIÓN DE FEATURES (X) Y TARGET (y)
    # ==========================================
    # Para que el modelo aprenda fisiología y clima, debemos evitar que identifique a cada parcela.
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

def entrenar_isolation_forest(X_if_scaled):
    """
    Entrena el modelo no supervisado para detectar anomalías agronómicas 
    basado exclusivamente en la distribución matemática de los datos crudos.
    """
    print("Iniciando Paso 2: Entrenamiento de Isolation Forest (No Supervisado)...\n")
    
    # ==========================================
    # 1. CONFIGURACIÓN DE HIPERPARÁMETROS
    # ==========================================
    # n_estimators: Cantidad de "árboles" que intentarán aislar los datos. 100 es el estándar óptimo.
    # contamination: Es la estimación de la tasa de anomalías en el mundo real. 
    #                Asumimos agronómicamente que alrededor del 10% (0.10) de los lotes no etiquetados 
    #                podrían estar sufriendo un evento extremo (plaga severa, sequía crítica).
    # random_state: Fijamos una semilla (42) para que el entrenamiento sea reproducible en la evaluación.
    # n_jobs=-1: Le indicamos al código que utilice todos los núcleos del procesador para mayor velocidad.
    
    modelo_if = IsolationForest(
        n_estimators=100, 
        contamination=0.10, 
        random_state=42, 
        n_jobs=-1
    )
    
    # ==========================================
    # 2. ENTRENAMIENTO (FIT)
    # ==========================================
    print("Construyendo la frontera matemática de normalidad fenológica...")
    # Atención: Usamos solo .fit() pasándole EXCLUSIVAMENTE las variables predictoras (X_if_scaled).
    # No le pasamos ninguna 'y' (etiquetas) porque el modelo debe descubrir los patrones por sí solo.
    modelo_if.fit(X_if_scaled)
    
    # ==========================================
    # 3. GUARDADO DEL MODELO ENTRENADO
    # ==========================================
    # Exportamos el "cerebro" ya entrenado del modelo a la carpeta models/
    # Este archivo .pkl es el que utilizará el ejecutable final para escanear nuevas imágenes.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_modelo = MODELS_DIR / "isolation_forest.pkl"
    joblib.dump(modelo_if, ruta_modelo)
    
    print(f"-> ¡Detector de Anomalías entrenado y guardado con éxito en: {ruta_modelo}!\n")
    
    return modelo_if

def entrenar_random_forest_base(X_train_scaled, y_train, X_val_scaled, y_val):
    """
    Entrena un modelo Random Forest de aprendizaje supervisado utilizando la 
    configuración por defecto para establecer una métrica de rendimiento inicial.
    """
    print("Iniciando Paso 3: Entrenamiento Base del Random Forest Classifier...\n")
    
    # ==========================================
    # 1. INSTANCIACIÓN DEL MODELO BASE
    # ==========================================
    # Inicializo el algoritmo con sus parámetros predeterminadas, con dos excepciones:
    # 1. random_state=42: Fija la semilla matemática para que los resultados sean reproducibles.
    # 2. class_weight='balanced': Estrategia de penalización. Como tengo menos fotos de 'Plaga' (117) 
    #    que de 'Baja_Densidad' (487), el modelo le dará más peso matemático a las clases minoritarias 
    #    durante el aprendizaje, evitando que se vuelva sesgado hacia las clases con más registros.
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # ==========================================
    # 2. ENTRENAMIENTO (APRENDIZAJE SUPERVISADO)
    # ==========================================
    print("Enseñando al algoritmo a relacionar índices y clima con las etiquetas agronómicas...")
    
    # El algoritmo toma las variables predictoras (X_train) 
    # y las respuestas correctas (y_train), y construye los árboles de decisión internos 
    # buscando las reglas matemáticas que asocian cada entrada con su resultado.
    rf_base.fit(X_train_scaled, y_train)
    
    # ==========================================
    # 3. PRIMERA EVALUACIÓN (CONJUNTO DE VALIDACIÓN)
    # ==========================================
    print("Evaluando el modelo base con el conjunto de Validación (14%)...")
    
    # Le pasamos el examen de validación (X_val). El modelo no conoce las respuestas de esto.
    y_pred_val = rf_base.predict(X_val_scaled)
    
    # Comparamos las respuestas que dio el modelo (y_pred_val) contra la realidad validada en campo (y_val)
    print("\n--- Reporte de Clasificación Base (Conjunto de Validación) ---")
    # Genero una tabla con Precisión, Recall y F1-Score para cada una de las clases
    print(classification_report(y_val, y_pred_val))
    
    # ==========================================
    # 4. GUARDADO DEL MODELO BASE
    # ==========================================
    # Guardo esta versión inicial por si necesito comparar métricas en el futuro
    ruta_modelo_base = MODELS_DIR / "rf_base.pkl"
    joblib.dump(rf_base, ruta_modelo_base)
    print(f"-> Modelo Base entrenado y guardado en: {ruta_modelo_base}\n")
    
    return rf_base

def optimizar_random_forest(X_train_scaled, y_train, X_val_scaled, y_val):
    """
    Aplico GridSearchCV para explorar sistemáticamente múltiples hiperparámetros.
    El objetivo es encontrar la combinación que maximice la precisión mientras 
    se penaliza la complejidad excesiva (prevención de overfitting).
    """
    print("Iniciando Paso 4: Optimización de Hiperparámetros (Grid Search)...\n")
    
    # ==========================================
    # 1. DEFINICIÓN DE LA CUADRÍCULA (HIPERESPACIO DE BÚSQUEDA)
    # ==========================================
    # Definimos los límites matemáticos que el algoritmo tiene permitido explorar.
    param_grid = {
        # n_estimators: Cantidad de árboles. Más árboles dan estabilidad, pero consumen más memoria.
        'n_estimators': [100, 200, 300],
        
        # max_depth: PROFUNDIDAD MÁXIMA. Este es el freno  contra el overfitting.
        # Evita que el árbol crezca infinitamente hasta memorizar una sola fila de datos.
        'max_depth': [10, 15, 20, None],
        
        # min_samples_split: Cantidad mínima de parcelas requeridas para dividir un nodo (crear una regla nueva).
        'min_samples_split': [2, 5, 10],
        
        # min_samples_leaf: Cantidad mínima de parcelas que deben quedar en la decisión final (la hoja del árbol).
        # Subir esto evita que el modelo cree reglas matemáticas aisladas para casos atípicos.
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Espacio de búsqueda definido. Entrenando múltiples arquitecturas de Random Forest...")
    
    # ==========================================
    # 2. INSTANCIACIÓN Y EJECUCIÓN DEL GRID SEARCH
    # ==========================================
    # Inicializo un Random Forest base manteniendo el balanceo de clases (necesario para las plagas porque son menos registros)
    rf_template = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # GridSearchCV automatiza el proceso de prueba y error. 
    # cv=5 significa "Validación Cruzada Interna de 5 pliegues", lo que añade una capa extra 
    # de rigor estadístico dividiendo el X_train en 5 subgrupos durante la búsqueda.
    # n_jobs=-1 usa todos los núcleos de la computadora para acelerar el cálculo.
    grid_search = GridSearchCV(
        estimator=rf_template,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1_macro', # Priorizo el F1-Score equilibrado entre todas las clases
        verbose=1 # Solo muestra una barra de progreso en la consola
    )
    
    # Ejecutamos la búsqueda masiva.
    grid_search.fit(X_train_scaled, y_train)
    
    # ==========================================
    # 3. EXTRACCIÓN DEL MODELO GANADOR
    # ==========================================
    # Extraemos el modelo con la mejor combinación matemática encontrada
    mejor_rf = grid_search.best_estimator_
    
    print("\n¡Búsqueda finalizada!")
    print(f"Los hiperparámetros óptimos encontrados son:\n{grid_search.best_params_}\n")
    
    # ==========================================
    # 4. EVALUACIÓN DEL MODELO OPTIMIZADO VS CONJUNTO DE VALIDACIÓN
    # ==========================================
    print("Sometiendo el modelo optimizado al examen de Validación (14%)...")
    # Hacemos que el  modelo prediga las respuestas del conjunto de validación
    y_pred_optimizado = mejor_rf.predict(X_val_scaled)
    
    print("\n--- Reporte de Clasificación OPTIMIZADO (Conjunto de Validación) ---")
    print(classification_report(y_val, y_pred_optimizado))
    
    # ==========================================
    # 5. GUARDADO DEL MODELO FINAL DE PRODUCCIÓN
    # ==========================================
    # Este es el archivo definitivo que usaremos para la fase de Test (Regla de Kalena) 
    # y para el software .exe final.
    ruta_modelo_optimizado = MODELS_DIR / "rf_optimizado.pkl"
    joblib.dump(mejor_rf, ruta_modelo_optimizado)
    print(f"-> Modelo optimizado guardado con éxito en: {ruta_modelo_optimizado}\n")
    
    return mejor_rf

# Bloque de ejecución de prueba.
if __name__ == "__main__":
    # Cargo y escalo los datos
    datos = preparar_datos()

    # Si los datos se cargaron correctamente, procedemos a entrar los modelos de machine learning
    if datos is not None:
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, X_if_scaled, features = datos
        
        # Entreno el modelo isolation forest pasándole solo la matriz no etiquetada
        modelo_anomalias = entrenar_isolation_forest(X_if_scaled)

        # Entreno el Clasificador Base necesario para el modelo random forest (Supervisado)
        modelo_rf_base = entrenar_random_forest_base(X_train_scaled, y_train, X_val_scaled, y_val)

        # Ejecutamos la optimización para exprimir el rendimiento sin sobreajustar
        modelo_rf_optimizado = optimizar_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)