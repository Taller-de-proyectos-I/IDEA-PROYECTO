# ==============================================================================
# Script de prueba para Asistente IA (DeepSeek)
#
# Copyright (c) 2025 Yersson Calderon Romero. Todos los derechos reservados.
#
# Autor: Yersson Calderon Romero
# ==============================================================================

from openai import OpenAI

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hola, mi nombres es Jherzon y me dedico a entrenar modelos de CNN"},
    ],
    stream=False
)

print(response.choices[0].message.content)

# ==============================================================================
# Fin del script de prueba.
# ==============================================================================
