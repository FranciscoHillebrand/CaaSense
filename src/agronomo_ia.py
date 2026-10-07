import os
import sys
# Forzamos la codificación UTF-8 en la salida de la consola de Windows
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from pathlib import Path
import ollama
# Importamos tu parser (Asegurate de que llm_data_parser.py esté en la misma carpeta src/)
from llm_data_parser import construir_contexto_llm

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
# Apuntamos a la carpeta donde se guardan los resultados del modelo con SMOTE
ITER_DIR = BASE_DIR / "reports" / "figures_smote" / "iteracion_1"

def extraer_datos_auditoria():
    """Lee el archivo txt del Isolation Forest y extrae el porcentaje de anomalía por parcela."""
    ruta_if = ITER_DIR / "auditoria_isolation_forest.txt"
    datos_if = {}
    
    if not ruta_if.exists():
        print(f"Error: No se encontró el archivo de auditoría IF en {ruta_if}")
        return datos_if

    with open(ruta_if, "r", encoding="utf-8") as f:
        lineas = f.readlines()
        
    parcela_actual = None
    diagnostico_actual = None
    
    for linea in lineas:
        if linea.startswith("Parcela:"):
            parcela_actual = linea.split(":")[1].strip()
        elif linea.startswith("Diagnóstico general del Agrónomo:"):
            diagnostico_actual = linea.split(":")[1].strip()
        elif linea.startswith("-> Píxeles Anómalos detectados:"):
            # Extraemos el porcentaje que está entre paréntesis ej: "1082.0 (37.18%)"
            try:
                porcentaje_str = linea.split("(")[1].split("%")[0]
                porcentaje = float(porcentaje_str)
                # Guardamos usando una tupla (parcela, diagnostico) como clave
                datos_if[(parcela_actual, diagnostico_actual)] = porcentaje
            except IndexError:
                pass
                
    return datos_if

def obtener_contexto_meteorologico(parcela, diagnostico_rf, fecha=None):
    """
    Busca en el CSV de testeo las variables climáticas reales asociadas a esa parcela y fecha.
    Lista para conectarse con el selector de fechas del MVP.
    """
    ruta_test = PROCESSED_DIR / "test_real.csv"
    if not ruta_test.exists():
        return "Datos meteorológicos no disponibles."
        
    df_test = pd.read_csv(ruta_test)
    
    # Filtramos por parcela (y por fecha si el usuario la selecciona en el MVP)
    if fecha and 'fecha' in df_test.columns:
        filtro = df_test[(df_test['parcela'] == parcela) & (df_test['fecha'] == fecha)]
    else:
        # Para la prueba actual, filtramos por parcela y diagnóstico tomando un registro representativo
        filtro = df_test[(df_test['parcela'] == parcela) & (df_test['etiqueta'] == diagnostico_rf)]
        
    if filtro.empty:
        return "Sin registros meteorológicos específicos para esta fecha."
        
    fila = filtro.iloc[0]
    fecha_registro = fila.get('fecha', 'Fecha no especificada')
    
    # Identificamos dinámicamente las columnas de clima (excluyendo metadatos e índices satelitales)
    indices_satelitales = ['ndvi', 'evi', 'savi', 'ndre','mtci','ndmi','msi',   'nbr']
    columnas_excluir = ['fecha', 'parcela', 'ubicacion_meteo', 'etiqueta']
    
    datos_clima = []
    for col in df_test.columns:
        if col not in columnas_excluir and not any(ind in col.lower() for ind in indices_satelitales):
            valor = fila[col]
            if isinstance(valor, (int, float)):
                datos_clima.append(f"{col}: {round(valor, 2)}")
                
    clima_str = ", ".join(datos_clima) if datos_clima else "Variables estándar de la estación."
    return f"Fecha de captura: {fecha_registro} | Condiciones meteorológicas del período: {clima_str}"

def generar_recomendacion_agronomica(parcela, diagnostico_rf, porcentaje_anomalia_if, fecha=None):
    """Conecta con Ollama (Gemma 3) para generar el diagnóstico."""
    print(f"\n[{parcela}] Consultando al Ingeniero Agrónomo IA (Gemma 3)...")
    
    # 1. Obtenemos el contexto de los modelos (Paso 2) y el contexto climático (Paso 3)
    contexto_ia = construir_contexto_llm(parcela, diagnostico_rf, porcentaje_anomalia_if)
    contexto_clima = obtener_contexto_meteorologico(parcela, diagnostico_rf, fecha)

    
    # System Prompt: Le damos la personalidad y las reglas estrictas
    system_prompt = """
    Eres un ingeniero agrónomo senior especializado en el cultivo de yerba mate en Misiones, Argentina.
    Tu tarea es redactar un informe diagnóstico corto, profesional y estructurado.
    
    REGLAS:
    1. Usa español rioplatense formal y preciso.
    2. Usa el formato de viñetas (*).
    3. Nunca inventes datos numéricos que no estén en el contexto provisto.
    4. Explica brevemente qué significa el diagnóstico desde la fisiología de la yerba mate.
    5. NO incluyas fechas, saludos, introducciones ni encabezados. Ve directamente a los 3 puntos solicitados.
    6. PROHIBIDO explicar qué es Random Forest, Isolation Forest o cómo funciona la inteligencia artificial. Hablá directamente del estado del yerbal.
    7. REALIDAD PRODUCTIVA (PERTINENCIA AGRONÓMICA): El cultivo de yerba mate en Misiones es a secano (sin riego artificial). JAMÁS recomiendes instalar riego por goteo, regar las plantas, ni hacer mediciones de laboratorio complejas (como potencial de turgencia).
    8. Tus recomendaciones de manejo deben basarse en prácticas reales de chacra: control de malezas, protección del suelo, monitoreo de plagas locales (rulo/psílido y kiritó/taladro grande), fertilización correctiva o regulación de la cosecha (tarefa/poda).
    9. SÉ BREVE Y DIRECTO: Escribí un máximo de 2 a 3 oraciones concisas por cada punto.
    """
    
    # User Prompt: Le pasamos el contexto generado por nuestro parser
    user_prompt = f"""
    Analizá los siguientes datos de la parcela y emití el diagnóstico cruzando el estado de la planta con las condiciones meteorológicas:

    - DATOS SATELITALES: {contexto_ia}
    - DATOS METEOROLÓGICOS: {contexto_clima}

    Respondé ÚNICAMENTE completando estos 3 apartados con viñetas (*):
    * Diagnóstico General: (Estado fisiológico actual del yerbal y cómo influyó el clima del período).
    * Análisis Espacial: (Interpretación práctica del porcentaje de superficie afectada en el lote: si es un problema uniforme o focalizado).
    * Plan de Acción Inmediato: (1 o 2 tareas concretas y realistas de campo para el productor).
    """
    
    try:
        response = ollama.chat(
            model='gemma3:4b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.1, 'top_p': 0.9} # Temperatura baja para que sea analítico y no creativo
        )
        
        print("\n" + "="*65)
        print(f" REPORTE TÉCNICO: {parcela}")
        print(f" ({contexto_clima})")
        print("="*65)
        print(response['message']['content'].strip())
        print("="*65 + "\n")
        
    except Exception as e:
        print(f"❌ Error al conectar con Ollama: {e}")

def ejecutar_pipeline_llm():
    print("Iniciando Paso 3: Prompt Engineering e Inyección Meteorológica...")
    datos_anomalias = extraer_datos_auditoria()
    
    if not datos_anomalias:
        return
        
    casos_prueba = [
        ("Kalena_Parcela_5", "0_Sano_Normal"),
        ("HelechosIII_Parcela_3_2", "4_Plaga_Activa")
    ]
    
    for parcela, diagnostico in casos_prueba:
        if (parcela, diagnostico) in datos_anomalias:
            porcentaje_if = datos_anomalias[(parcela, diagnostico)]
            generar_recomendacion_agronomica(parcela, diagnostico, porcentaje_if)

if __name__ == "__main__":
    ejecutar_pipeline_llm()