# Please install OpenAI SDK first: `pip3 install openai`
from openai import OpenAI

client = OpenAI(api_key="sk-0f2c39c1ce7149f3905729653f00c1af", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hola, mi nombres es Jherzon y me dedico a entrenar modelos de CNN"},
    ],
    stream=False
)

print(response.choices[0].message.content)

