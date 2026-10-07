def limpiar_etiqueta(etiqueta_cruda):
    """Limpia los prefijos numéricos de las clases (ej: '4_Plaga_Activa' -> 'Plaga Activa')"""
    # Si la etiqueta viene con el formato "Numero_Texto", nos quedamos con la parte del texto
    if '_' in etiqueta_cruda and etiqueta_cruda[0].isdigit():
        partes = etiqueta_cruda.split('_', 1)[1]
        return partes.replace('_', ' ')
    return etiqueta_cruda.replace('_', ' ')

def construir_contexto_llm(parcela, diagnostico_rf, porcentaje_anomalia_if):
    """
    Traduce los resultados matemáticos fríos en un párrafo de contexto narrativo 
    para inyectar en el System Prompt de Gemma3:4b.
    """
    diagnostico_limpio = limpiar_etiqueta(diagnostico_rf)
    
    # 1. Base del contexto indicando la parcela y la predicción principal (Random Forest)
    contexto = f"El sistema de aprendizaje supervisado ha clasificado la parcela '{parcela}' con un diagnóstico principal de {diagnostico_limpio}. "
    
    # 2. Lógica condicional de la Auditoría Espacial (Isolation Forest)
    # Regla estricta: Menos de 5% se considera estabilizado/homogéneo
    if porcentaje_anomalia_if < 5.0:
        contexto += (
            f"El análisis de variabilidad espacial indica que el lote se encuentra estabilizado y homogéneo, "
            f"presentando apenas un {porcentaje_anomalia_if}% de píxeles atípicos. "
            "No se observan focos de alteración significativos dentro de la parcela."
        )
    # Entre 5% y 15%: Focos incipientes
    elif 5.0 <= porcentaje_anomalia_if <= 15.0:
        contexto += (
            f"La auditoría espacial ha detectado una anomalía leve del {porcentaje_anomalia_if}% de la superficie. "
            "Esto sugiere la presencia de focos incipientes o alteraciones tempranas muy localizadas que requieren monitoreo."
        )
    # Mayor a 15%: Dispersión significativa
    else:
        contexto += (
            f"La auditoría espacial revela una variabilidad crítica, con un {porcentaje_anomalia_if}% de la superficie "
            "marcada como anómala. Esto indica una dispersión significativa del problema o un estado de afectación generalizado."
        )
        
    return contexto

# Bloque de prueba para verificar cómo se generan los textos
if __name__ == "__main__":
    print("--- Prueba del Parser para el LLM ---")
    
    # Caso 1: Plaga con baja anomalía (Falso positivo o plaga homogeneizada)
    ctx1 = construir_contexto_llm("Kalena_Parcela_5", "4_Plaga_Activa", 2.76)
    print(f"\nCaso 1:\n{ctx1}")
    
    # Caso 2: Estrés Hídrico con alta anomalía (Confirmación del modelo dual)
    ctx2 = construir_contexto_llm("Basilio_Parcela_10", "1_Estres_Hidrico", 68.76)
    print(f"\nCaso 2:\n{ctx2}")