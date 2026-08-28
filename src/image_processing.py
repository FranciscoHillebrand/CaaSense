import os
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

# ==========================================
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# ==========================================
# BASE_DIR calcula la ruta absoluta de la carpeta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
# RAW_DIR es la ruta donde están guardadas las imágenes satelitales originales
RAW_DIR = BASE_DIR / "data" / "raw" / "Dataset archivos tiff"
# PROCESSED_DIR es la ruta donde se guardará el archivo CSV final con los resultados
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ==========================================
# DICCIONARIO TRADUCTOR DE PARCELAS
# ==========================================
# Este diccionario sirve para identificar a qué parcela corresponde cada imagen 
# basándose en el sufijo final del nombre de la carpeta de fecha (ej: "01-08-2022-2")
MAPEO_ETIQUETADAS = {
    "": "Kalena_Parcela_1",
    "-2": "Kalena_Parcela_2",
    "-3": "Kalena_Parcela_3",
    "-4": "Kalena_Parcela_4",
    "-5": "Kalena_Parcela_5",
    "-6": "Kalena_Parcela_6",
    "-7": "Basilio_Parcela_7",
    "-8": "Basilio_Parcela_8",
    "-9": "Basilio_Parcela_9",
    "-10": "Basilio_Parcela_10",
    "-11": "Basilio_Parcela_11",
    "-12": "Basilio_Parcela_12",
    "-13": "Basilio_Parcela_13",
    "-14": "Basilio_Parcela_14",
    "-15": "Basilio_Parcela_15",
    "-16": "Basilio_Parcela_16",
    "-1-1 Los Helechos I": "HelechosI_Parcela_1_1",
    "-1-2 Los Helechos I": "HelechosI_Parcela_1_2",
    "-1-3 Los Helechos I": "HelechosI_Parcela_1_3",
    "-1-4 Los Helechos I": "HelechosI_Parcela_1_4",
    "-2-1 Los Helechos II": "HelechosII_Parcela_2_1",
    "-2-2 Los Helechos II": "HelechosII_Parcela_2_2",
    "-3-1 Los Helechos III": "HelechosIII_Parcela_3_1",
    "-3-2 Los Helechos III": "HelechosIII_Parcela_3_2",
    "-4-5 Jardín América": "Jardin_America_Parcela_4_5",
    "-6-1 Caraguatay": "Caraguatay_Parcela_6_1",
    "-6-2 Caraguatay": "Caraguatay_Parcela_6_2",
    "-6-3 Caraguatay": "Caraguatay_Parcela_6_3"
}

def leer_banda(ruta_carpeta: Path, patron: str) -> np.ndarray:
    """Busca un archivo TIFF por un patrón de texto en el nombre (ej. 'B04') y lo lee."""
    # Busca cualquier archivo en la carpeta que contenga el patrón en su nombre
    archivos = list(ruta_carpeta.glob(f"*{patron}*.tiff"))
    # Si no encuentra el archivo (ej. falta una banda), devuelve None para no frenar el programa
    if not archivos:
        return None
    # Abre la imagen satelital y extrae la matriz de píxeles convirtiéndola a números decimales
    with rasterio.open(archivos[0]) as src:
        return src.read(1).astype(np.float32)

def procesar_carpeta_fecha(ruta_fecha: Path) -> dict:
    """Lee las bandas de una fecha específica, aplica la máscara de nubes y calcula los índices."""
    
    # 1. Lectura de bandas satelitales (Colores y Espectros invisibles)
    b02 = leer_banda(ruta_fecha, "B02") # Azul
    b04 = leer_banda(ruta_fecha, "B04") # Rojo
    b05 = leer_banda(ruta_fecha, "B05") # Borde Rojo (Red Edge)
    b08 = leer_banda(ruta_fecha, "B08") # Infrarrojo Cercano (NIR)
    b11 = leer_banda(ruta_fecha, "B11") # Infrarrojo de Onda Corta 1 (SWIR)
    b12 = leer_banda(ruta_fecha, "B12") # Infrarrojo de Onda Corta 2 (SWIR)

    # Lee el archivo que indica qué píxeles son válidos y cuáles son nubes o vacíos
    mascara = leer_banda(ruta_fecha, "mask")

    # Si faltan las bandas críticas (Rojo e Infrarrojo), es imposible calcular índices básicos, se cancela.
    if b04 is None or b08 is None:
        return None

    # 2. Enmascarado de nubes (Cloud Masking)
    if mascara is not None:
        # Crea una condición donde solo son válidos los píxeles que en la máscara valen 1
        condicion_valida = mascara == 1
        # np.where reemplaza los píxeles inválidos (nubes) por 'np.nan' (Sin Dato)
        # Esto asegura que las nubes no contaminen los cálculos matemáticos
        b02 = np.where(condicion_valida, b02, np.nan) if b02 is not None else None
        b04 = np.where(condicion_valida, b04, np.nan)
        b05 = np.where(condicion_valida, b05, np.nan) if b05 is not None else None
        b08 = np.where(condicion_valida, b08, np.nan)
        b11 = np.where(condicion_valida, b11, np.nan) if b11 is not None else None
        b12 = np.where(condicion_valida, b12, np.nan) if b12 is not None else None
    
    
    # 3. Cálculo de Índices Espectrales (Método Académico)
    resultados = {}

    def calcular_indice(numerador, denominador):
        """Calcula el índice devolviendo NaN (Sin dato) donde el denominador sea 0 para evitar errores."""
        # np.divide realiza la división de matrices de forma segura. 
        # Si encuentra un cero en el denominador, inyecta np.nan sin romper el programa ni alterar la fórmula
        return np.divide(
            numerador,
            denominador,
            out=np.full_like(denominador, np.nan, dtype=np.float32),
            where=(denominador != 0)
        )

    # NDVI (Índice de Vegetación de Diferencia Normalizada)
    ndvi = calcular_indice(b08 - b04, b08 + b04)
    # Guardamos la mediana de la parcela para ignorar valores atípicos
    resultados['NDVI'] = np.nanmedian(ndvi)
    
    # EVI (Índice de Vegetación Mejorado) - Corrige efectos atmosféricos
    if b02 is not None:
        evi = calcular_indice(2.5 * (b08 - b04), b08 + 6 * b04 - 7.5 * b02 + 1)
        resultados['EVI'] = np.nanmedian(evi)
        
    # SAVI (Índice de Vegetación Ajustado al Suelo)
    # L=0.5 factor precalculado que minimiza el brillo del suelo desnudo
    savi = calcular_indice((b08 - b04) * 1.5, b08 + b04 + 0.5)
    resultados['SAVI'] = np.nanmedian(savi)
    
    if b05 is not None:
        # NDRE (Índice de Diferencia Normalizada de Borde Rojo) - Sensible a clorofila
        ndre = calcular_indice(b08 - b05, b08 + b05)
        resultados['NDRE'] = np.nanmedian(ndre)
        # MTCI (Índice de Clorofila Terrestre de MERIS)
        mtci = calcular_indice(b08 - b05, b05 - b04)
        resultados['MTCI'] = np.nanmedian(mtci)
        
    if b11 is not None:
        # NDMI (Índice de Diferencia Normalizada de Humedad) - Detecta estrés hídrico
        ndmi = calcular_indice(b08 - b11, b08 + b11)
        resultados['NDMI'] = np.nanmedian(ndmi)
        # MSI (Índice de Estrés por Humedad) - Relación inversa, mayor valor = mayor sequía
        msi = calcular_indice(b11, b08)
        resultados['MSI'] = np.nanmedian(msi)
        
    if b12 is not None:
        # NBR (Índice de Área Quemada Normalizada) - Útil para daño severo en tejidos
        nbr = calcular_indice(b08 - b12, b08 + b12)
        resultados['NBR'] = np.nanmedian(nbr)
        
    return resultados

def extraer_datos_satelitales(directorio_base: Path):
    """Recorre toda la estructura de carpetas, procesa las imágenes y genera un CSV."""
    registros = []
    print(f"Iniciando escaneo de imágenes en: {directorio_base}...\n")

    # Busca recursivamente todos los archivos que terminen en .tiff sin importar la profundidad
    archivos_tiff = list(directorio_base.rglob("*.tiff"))

    # Extrae solo las carpetas únicas que contienen imágenes para no procesar el mismo lugar dos veces
    carpetas_con_imagenes = set([archivo.parent for archivo in archivos_tiff])
    
    for ruta_fecha in carpetas_con_imagenes:
        # Divide la ruta en sus partes individuales (carpetas)
        partes = ruta_fecha.parts
        
        try:
            # Encuentra dónde empieza nuestra base de datos para leer la estructura
            idx_base = partes.index("Dataset archivos tiff")
            subruta = partes[idx_base+1:] 
        except ValueError:
            continue

        # El último elemento de la subruta es siempre la carpeta que contiene la fecha
        nombre_carpeta_fecha = subruta[-1]
        # Extrae los primeros 10 caracteres que corresponden a la fecha real (dd-mm-yyyy)
        fecha_real = nombre_carpeta_fecha[:10] 

        # Lógica para extraer la clasificación y el nombre de las parcelas SIN etiqueta
        if subruta[0] == "01_No_Etiquetadas":
            etiqueta = "No_Etiquetada"
            # Asume que la parcela es la carpeta anterior a la fecha
            parcela = subruta[-2] if len(subruta) >= 3 else "Desconocida"

        # Lógica para extraer la clasificación y el nombre de las parcelas CON etiqueta
        elif subruta[0] == "02_Etiquetadas":
            # Extrae la etiqueta (ej. 4_Plaga_Activa)
            etiqueta = subruta[1] if len(subruta) >= 2 else "Etiquetada_Generica"
            # Extrae el sufijo restándole los primeros 10 caracteres de la fecha
            sufijo = nombre_carpeta_fecha[10:]
            # Usa el diccionario traductor para buscar el nombre real de la parcela según su sufijo
            parcela = MAPEO_ETIQUETADAS.get(sufijo, f"Desconocida ({sufijo})")
        else:
            continue
        
        # Imprime en consola el progreso
        print(f"Procesando: {parcela} | Fecha: {fecha_real} | [{etiqueta}]")
        # Llama a la función matemática para procesar la imagen actual
        indices = procesar_carpeta_fecha(ruta_fecha)
        
        # Si se procesó con éxito, agrega los metadatos y lo suma a la lista de registros
        if indices:
            indices.update({"fecha": fecha_real, "parcela": parcela, "etiqueta": etiqueta})
            registros.append(indices)

    # Convierte la lista completa de registros en una tabla de datos (DataFrame)
    df = pd.DataFrame(registros)

    if not df.empty:
        # Crea la carpeta de destino si no existe
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Estandariza la columna de fechas para que Pandas la reconozca como fecha oficial
        df['fecha'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y', errors='coerce')

        # Define la ruta final y guarda la tabla como un archivo CSV
        ruta_salida = PROCESSED_DIR / "indices_satelitales.csv"
        df.to_csv(ruta_salida, index=False)
        print(f"\n¡Procesamiento finalizado! Dataset guardado en: {ruta_salida} con {len(df)} registros.")
    else:
        print("\nNo se encontraron imágenes válidas para procesar.")

# Este bloque asegura que el código solo se ejecute si se lanza directamente este archivo
if __name__ == "__main__":
    extraer_datos_satelitales(RAW_DIR)