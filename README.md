# Sistema de Recomendación Inteligente para Yerba Mate

## Descripción del Proyecto
Este proyecto consiste en el desarrollo de un prototipo de software de recomendación inteligente que sirve de soporte para la toma de decisiones agronómicas en plantaciones de yerba mate. El sistema procesa y analiza imágenes satelitales multiespectrales (Sentinel-2) y datos meteorológicos (Open-Meteo) para identificar anomalías de forma temprana. 

Su característica principal es la integración de Modelos de Lenguaje Grandes (LLMs) ejecutados localmente (Ollama) para traducir los datos numéricos y espectrales en recomendaciones agronómicas personalizadas y expresadas en lenguaje natural.

## Arquitectura y Estrategia de Datos
Para sortear las limitaciones de cuota de las APIs satelitales comerciales (Copernicus) y garantizar un rendimiento óptimo en la aplicación de escritorio final, la estrategia de datos se divide en dos enfoques:
*   **Fase de Entrenamiento y Modelado:** Utiliza un dataset histórico estático compuesto por archivos `.tiff` recortados y validados, almacenados localmente.
*   **Fase de Producción (Prototipo final):** El sistema procesará imágenes individuales ingresadas por el usuario y consultará el clima de los últimos 30 días bajo demanda para emitir diagnósticos en tiempo real.

## Estructura del Proyecto
El repositorio está organizado en un formato modular para separar claramente los datos crudos del código fuente:

```text
Sistema_Recomendacion_Yerba/
│
├── data/                   # Almacenamiento local de información (No subir a Git)
│   ├── raw/                # Dataset histórico de imágenes satelitales (.tiff)
│   ├── meteo/              # Registros meteorológicos descargados (.csv)
│   └── processed/          # Dataset unificado para entrenamiento de ML
│
├── src/                    # Código fuente del backend y lógica principal
│   ├── data_ingestion.py   # Scripts de conexión a Open-Meteo y lectura de archivos
│   ├── image_processing.py # Cálculo de índices espectrales (NDVI, NDRE, NDWI, etc.)
│   └── ml_models.py        # Entrenamiento y ejecución de Isolation/Random Forest
│
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación general

## Configuración del Entorno (Setup)

1. **Crear el entorno virtual:**
   ```bash
   python -m venv venv

2. **Activar el entorno virtual (Windows):**
    ```bash
   .\venv\Scripts\activate

3. **Instalar las dependencias requeridas:**
   ```bash
   pip install -r requirements.txt
   

## Tecnologías Principales
*   **Procesamiento Espacial:** `rasterio`, `geopandas`, `numpy`
*   **Manipulación de Datos:** `pandas`, `requests`
*   **Machine Learning:** `scikit-learn` (Isolation Forest, Random Forest)
*   **IA Generativa:** `ollama` (Ejecución local de LLMs)
*   **Interfaz Gráfica:** `customtkinter`

## Estado Actual del Desarrollo (Roadmap)
- [x] Planificación arquitectónica y definición de estructura.
- [x] Configuración del entorno virtual e instalación de dependencias.
- [x] Ingesta manual del dataset histórico de imágenes (`data/raw/`).
- [ ] Refactorización y automatización de descarga de datos meteorológicos.
- [ ] Procesamiento de imágenes y cálculo de índices espectrales.