import os
from flask import Flask, flash, redirect, render_template, request, url_for
os.environ['PYTHONIOENCODING'] = 'utf-8'
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from flask_mail import Mail, Message
from email.header import Header
from deep_translator import GoogleTranslator

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_2.pt", map_location='cpu'))
model.eval()

def prediction(image_path, top_k=3):
    """
    Realiza una predicción sobre una imagen y devuelve las 'top_k' predicciones más probables.
    """
    image = Image.open(image_path)
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
app.secret_key = 'supersecretkey'

# --- Configuración de Flask-Mail ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'tu-correo@gmail.com'  # <-- CAMBIA ESTO por tu correo de envío
app.config['MAIL_PASSWORD'] = 'tu-contraseña-de-aplicacion'  # <-- CAMBIA ESTO por tu contraseña
app.config['MAIL_DEFAULT_SENDER'] = ('PlantAndes', app.config['MAIL_USERNAME'])

mail = Mail(app)

@app.route('/')
def home_page():
    return render_template('home.html') # This is now the main landing page

@app.route('/contact')
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message_body = request.form['message']

        subject = f"Nuevo Mensaje de Contacto de: {name}"
        msg = Message(subject=Header(subject, 'utf-8'),
                      recipients=['71827961@continental.edu.pe'],
                      reply_to=email)

        msg.html = render_template('email_template.html', name=name, email=email, message_body=message_body)
        mail.send(msg)
        
        flash('¡Gracias por tu mensaje! Nos pondremos en contacto contigo pronto.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact-us.html')

@app.route('/index')
@app.route('/index', methods=['GET', 'POST'])
def ai_engine_page():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image:
            # Si no se sube ninguna imagen, volver a renderizar la página
            return render_template('index.html', error="No se seleccionó ninguna imagen.")

        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        user_image_url = 'uploads/' + filename
        image.save(file_path)
        
        top_probs, top_indices = prediction(file_path)

        results = []
        for i in range(len(top_probs)):
            pred_index = top_indices[i]

            # Traducir la descripción y las recomendaciones al español
            description_es = GoogleTranslator(source='en', target='es').translate(disease_info['description'][pred_index])
            prevent_es = GoogleTranslator(source='en', target='es').translate(disease_info['Possible Steps'][pred_index])

            result = {
                'disease_name': disease_info['disease_name'][pred_index].replace("_", " "),
                'probability': round(top_probs[i] * 100, 2),
                'description': description_es,
                'prevent': prevent_es,
                'supplement_name': supplement_info['supplement name'][pred_index],
                'supplement_image': supplement_info['supplement image'][pred_index],
                'buy_link': supplement_info['buy link'][pred_index]
            }
            results.append(result)

        return render_template('index.html', results=results, user_image=user_image_url)

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
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
