import subprocess
import os
import sys
import time
from pathlib import Path
import pandas as pd

SCRIPTS = [
    "src/data_splitting.py",
    "src/extract_pixels_if.py",
    "src/if_data_fusion.py",
    "src/if_data_cleaning.py",
    "src/if_data_test.py",
    "src/ml_models.py",
    "src/evaluate_models.py"
]

def ejecutar_validacion_cruzada():
    print(f"Usando intérprete (venv): {sys.executable}\n")
    entorno = os.environ.copy()
    
    for iteracion in range(1, 6):
        print(f"\n{'='*50}\n INICIANDO ITERACIÓN {iteracion} DE 5\n{'='*50}")
        entorno["ITERACION_CV"] = str(iteracion)
        
        for script in SCRIPTS:
            print(f"-> Ejecutando {script}...")
            resultado = subprocess.run([sys.executable, script], env=entorno)
            
            if resultado.returncode != 0:
                print(f" Error en {script}. Abortando iteración.")
                return 
                
        print(f" Iteración {iteracion} completada.")
        time.sleep(2)

if __name__ == "__main__":
    ejecutar_validacion_cruzada()