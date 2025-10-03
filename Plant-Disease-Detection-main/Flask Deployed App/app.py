import os
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
os.environ['PYTHONIOENCODING'] = 'utf-8'
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import pickle
from flask_mail import Mail, Message
from email.header import Header
from deep_translator import GoogleTranslator

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

# --- Estandarización de Nombres de Enfermedades ---
disease_info['disease_name'] = disease_info['disease_name'].str.replace(' : ', '___', regex=False).str.replace(' ', '_', regex=False)

# --- Estandarización de Nombres de Columnas (PRE-CARGA) ---
supplement_info.columns = [col.replace(' ', '_') for col in supplement_info.columns]

# --- Filtrar la clase de fondo no deseada ---
background_class_name = 'Background_without_leaves'
if background_class_name in disease_info['disease_name'].values:
    # Obtener el índice de la fila a eliminar
    background_index = disease_info[disease_info['disease_name'] == background_class_name].index
    # Eliminar la fila de ambos DataFrames
    disease_info.drop(background_index, inplace=True)
    supplement_info.drop(background_index, inplace=True)

# --- CARGA EFICIENTE DE DATOS CON CACHÉ  ---
CACHE_FILE = 'translated_data.pkl'

cache_loaded = False
if os.path.exists(CACHE_FILE):
    try:
        print("Intentando cargar datos pre-traducidos desde el caché...")
        with open(CACHE_FILE, 'rb') as f:
            disease_info, supplement_info = pickle.load(f)
        print("Carga desde caché completada.")
        cache_loaded = True
    except (EOFError, pickle.UnpicklingError):
        print("ADVERTENCIA: El archivo de caché está corrupto. Se eliminará y se volverá a crear.")
        os.remove(CACHE_FILE)

if not cache_loaded:
    print("Caché no encontrado. Iniciando pre-traducción de datos (esto solo ocurrirá una vez)...")
    try:
        translator = GoogleTranslator(source='en', target='es')

        # Preparar textos para traducir
        disease_names_en = [name.replace("_", " ").replace("___", " - ") for name in disease_info['disease_name']]
        supplement_names_en = [name if pd.notna(name) else "" for name in supplement_info['supplement_name']]
        descriptions_en = [desc if pd.notna(desc) else "" for desc in disease_info['description']]
        steps_en = [step if pd.notna(step) else "" for step in disease_info['Possible Steps']]

        # Traducir en lotes
        disease_info['disease_name_es'] = translator.translate_batch(disease_names_en)
        supplement_info['supplement_name_es'] = translator.translate_batch(supplement_names_en)
        disease_info['description_es'] = translator.translate_batch(descriptions_en)
        disease_info['steps_es'] = translator.translate_batch(steps_en)

        # Guardar en caché para futuros inicios
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump((disease_info, supplement_info), f)
        
        print("Pre-traducción completada y guardada en caché.")
    except Exception as e:
        print(f"ERROR: Falló la pre-traducción. La aplicación podría no mostrar textos en español. Error: {e}")
        # En caso de error, creamos las columnas para que la app no se caiga
        for col in ['disease_name_es', 'description_es', 'steps_es', 'supplement_name_es']:
            if col not in disease_info.columns and col not in supplement_info.columns:
                if 'supplement' in col: supplement_info[col] = ""
                else: disease_info[col] = ""

model = CNN.CNN(39)    
# Forzar la carga en la CPU si no hay GPU disponible
model.load_state_dict(torch.load("plant_disease_model_2.pt", map_location='cpu'))
model.eval()

def prediction(image_path, top_k=3):
    """
    Realiza una predicción sobre una imagen y devuelve las 'top_k' predicciones más probables.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    
    with torch.no_grad():
        logits = model(input_data)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k)

    return top_probs[0].tolist(), top_indices[0].tolist()


app = Flask(__name__)

# Clave secreta para mensajes flash
app.secret_key = os.environ.get('SECRET_KEY', 'una-clave-secreta-por-defecto-para-desarrollo')

# --- Configuración de Flask-Mail ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = ('PlantAndes', app.config['MAIL_USERNAME'])

mail = Mail(app)

@app.route('/')
def home_page():
    return render_template('home.html') # This is now the main landing page

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message_body = request.form['message']

        subject = f"Nuevo Mensaje de Contacto de: {name}"
        msg = Message(subject=Header(subject, 'utf-8'), # El destinatario podría ser una variable de entorno
                      recipients=['71827961@continental.edu.pe'],
                      reply_to=email)

        msg.html = render_template('email_template.html', name=name, email=email, message_body=message_body)
        mail.send(msg)
        
        flash('¡Gracias por tu mensaje! Nos pondremos en contacto contigo pronto.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact-us.html')

@app.route('/index', methods=['GET', 'POST'])
def ai_engine_page():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image:
            flash("No se seleccionó ninguna imagen.", "error")
            return redirect(request.url)

        filename = secure_filename(image.filename)
        file_path = os.path.join('static/uploads', filename)
        # Asegurarse de que el directorio 'uploads' exista
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        image.save(file_path)
        
        top_probs, top_indices = prediction(file_path)

        results = []
        for i in range(len(top_probs)):
            pred_index = top_indices[i]

            # Usar los datos pre-traducidos del DataFrame para máxima eficiencia
            disease_name_es = disease_info.loc[pred_index, 'disease_name_es']
            description_es = disease_info.loc[pred_index, 'description_es']
            prevent_es = disease_info.loc[pred_index, 'steps_es']
            supplement_name_es = supplement_info.loc[pred_index, 'supplement_name_es']

            result = {
                'disease_name': disease_name_es,
                'probability': round(top_probs[i] * 100, 2),
                'description': description_es,
                'prevent': prevent_es,

                'supplement_name': supplement_name_es,
                'supplement_image': supplement_info.loc[pred_index, 'supplement_image'],
                'buy_link': supplement_info.loc[pred_index, 'buy_link']
            }
            results.append(result)

        return render_template('index.html', results=results, user_image=f"uploads/{filename}")

    # Para peticiones GET, simplemente renderiza la página
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    # La ruta ahora solo sirve los datos pre-traducidos, haciéndola instantánea.
    return render_template('market.html', 
                           diseases=disease_info.to_dict(orient='records'),
                           supplements=supplement_info.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
