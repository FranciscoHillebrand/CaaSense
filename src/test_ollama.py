import ollama
import time

print("Iniciando conexión con Ollama (gemma3:4b)...")

system_prompt = """
Eres un ingeniero agrónomo senior especializado en el cultivo de yerba mate en la provincia de Misiones, Argentina. 
Tus respuestas deben ser técnicas, precisas, en español formal, y enfocadas estrictamente en la fisiología de la yerba mate.
"""

user_prompt = "Explica en un solo párrafo cuáles son los primeros síntomas visuales del estrés hídrico en las hojas de la planta."

start_time = time.time()

try:
    response = ollama.chat(
        model='gemma3:4b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        options={
            'temperature': 0.2
        }
    )
    
    end_time = time.time()
    
    print("\n--- Respuesta de gemma3:4b ---")
    print(response['message']['content'])
    print("-------------------------------------------\n")
    print(f" ¡Prueba superada en {round(end_time - start_time, 2)} segundos!")
    
except Exception as e:
    print(f" Error al conectar: {e}")