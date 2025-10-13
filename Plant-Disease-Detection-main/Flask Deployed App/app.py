import os
from flask import Flask, flash, redirect, render_template, request, url_for, session, jsonify, Response
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
from openai import OpenAI # Para la IA Generativa
import requests # Importar la librería requests
# --- IMPORTACIONES DE MODELOS Y AUTENTICACIÓN ---
from models import db, User, Zone, Diagnosis 
from flask_migrate import Migrate
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

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

# --- CONFIGURACIÓN DE BASE DE DATOS ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plantandes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- INICIALIZAR EXTENSIONES ---
db.init_app(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirige a la página de login si se intenta acceder a una ruta protegida
login_manager.login_message = "Por favor, inicie sesión para acceder a esta página."
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Clave secreta para mensajes flash
app.secret_key = os.environ.get('SECRET_KEY', 'una-clave-secreta-por-defecto-para-desarrollo')

# --- Configuración de la API de IA Generativa (DeepSeek) ---
app.config['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY', '') # Reemplaza con tu clave o usa variables de entorno

ai_client = OpenAI(api_key=app.config['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")


# --- Configuración de Flask-Mail ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = ('PlantAndes', app.config['MAIL_USERNAME'])

mail = Mail(app)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home_page'))

# --- RUTA PARA EL ASISTENTE DE IA GENERATIVA ---
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    user_message = data.get('message')
    context = data.get('context')
    language = data.get('language', 'Español') # Obtener idioma, por defecto Español

    if not user_message or not context:
        return jsonify({'error': 'Falta mensaje o contexto'}), 400

    # Prompt de sistema para guiar a la IA
    system_prompt = (
        "Eres 'AndesGPT', un asistente experto en agronomía y fitopatología, especializado en cultivos andinos. "
        "Tu propósito es ayudar a los agricultores a entender y manejar las enfermedades de sus plantas. "
        "Actúa de forma amigable, profesional y da respuestas claras y accionables. "
        f"Utiliza el siguiente contexto para responder la pregunta del usuario. No inventes información. Responde siempre en el idioma: {language}."
    )

    # Combinar contexto y pregunta del usuario
    full_prompt = (
        f"Contexto del diagnóstico:\n"
        f"- Enfermedad detectada: {context.get('disease_name')}\n"
        f"- Probabilidad de diagnóstico: {context.get('probability')}%\n"
        f"- Descripción: {context.get('description')}\n"
        f"- Recomendaciones: {context.get('prevent')}\n\n"
        f"Pregunta del usuario: {user_message}"
    )

    def generate():
        try:
            stream = ai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"Error durante el streaming: {e}")
            yield "Lo siento, ocurrió un error al generar la respuesta."

    return Response(generate(), mimetype='text/plain')

@app.route('/general-chat', methods=['POST'])
def general_chat():
    """Maneja las consultas generales del chat flotante."""
    data = request.json
    user_message = data.get('message')
    language = data.get('language', 'Español')

    if not user_message:
        return jsonify({'error': 'Falta mensaje'}), 400

    system_prompt = (
        "Eres 'AndesGPT', un asistente experto en agronomía y fitopatología, especializado en cultivos andinos. "
        "Tu propósito es ayudar a los agricultores. Si te preguntan sobre una enfermedad específica, anímales a usar el 'Motor IA' "
        "y subir una foto para un diagnóstico preciso. Puedes responder preguntas generales sobre agricultura, "
        f"técnicas de cultivo, plagas comunes, etc. Responde siempre en el idioma: {language}."
    )

    def generate():
        try:
            stream = ai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                stream=True
            )
            for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    yield content
        except Exception as e:
            print(f"Error en el chat general: {e}")
            yield "Lo siento, ocurrió un error al conectar con el asistente."

    return Response(generate(), mimetype='text/plain')

@app.cli.command("seed-zones")
def seed_zones():
    """Pobla la base de datos con todos los distritos de Perú desde una API externa."""
    if Zone.query.first():
        print("Las zonas ya han sido añadidas anteriormente.")
        return

    API_URL = "https://free.e-api.net.pe/ubigeos.json"
    print(f"Obteniendo datos de ubicación desde {API_URL}...")

    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        ubigeos = response.json()

        new_zones = []

        # Nivel 1: Departamentos
        for dep_name, provincias in ubigeos.items():
            if not isinstance(provincias, dict):
                continue

            # Nivel 2: Provincias
            for prov_name, distritos in provincias.items():
                if not isinstance(distritos, dict):
                    continue

                # Nivel 3: Distritos
                for dist_name, dist_data in distritos.items():
                    if not isinstance(dist_data, dict):
                        continue

                    zone = Zone(
                        department_name=dep_name,
                        province_name=prov_name,
                        district_name=dist_name
                    )
                    new_zones.append(zone)

        if new_zones:
            db.session.bulk_save_objects(new_zones)
            db.session.commit()
            print(f"¡Base de datos poblada con {len(new_zones)} distritos de Perú!")
        else:
            print("No se encontraron distritos para guardar.")

    except requests.exceptions.RequestException as e:
        print(f"Error al obtener datos de la API: {e}")

@app.route('/')
def home_page():
    return render_template('home.html') # This is now the main landing page

# --- RUTAS DE AUTENTICACIÓN ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        zone_id = request.form.get('district') # El ID de la zona ahora viene del campo 'district'
        address = request.form.get('address')

        # Validaciones
        if User.query.filter_by(username=username).first():
            flash('El nombre de usuario ya existe.', 'warning')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('El correo electrónico ya está registrado.', 'warning')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, zone_id=zone_id, address=address)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('¡Registro exitoso! Ahora puedes iniciar sesión.', 'success')
        return redirect(url_for('login'))

    # Preparar datos para los desplegables dinámicos
    all_zones_objects = Zone.query.order_by(Zone.department_name, Zone.province_name, Zone.district_name).all()
    
    # Obtenemos una lista única y ordenada de departamentos
    departments = sorted(list(set(z.department_name for z in all_zones_objects)))
    
    # Convertimos la lista de objetos a una lista de diccionarios (JSON serializable)
    all_zones_serializable = [{'id': z.id, 'district_name': z.district_name, 'province_name': z.province_name, 'department_name': z.department_name} for z in all_zones_objects]
    
    return render_template('register.html', departments=departments, all_zones=all_zones_serializable)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home_page'))
        else:
            flash('Credenciales inválidas. Por favor, inténtalo de nuevo.', 'danger')

    return render_template('login.html')

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
        
        # --- LÓGICA DE VALIDACIÓN DE CALIDAD (HU-18) ---
        # Aquí iría tu función para chequear si la imagen es borrosa.
        # from image_validator import check_blur
        # quality_score, is_blurry = check_blur(file_path)
        # if is_blurry:
        #     flash(f"La imagen parece borrosa (Calidad: {quality_score:.2f}). Por favor, sube una imagen más nítida.", "warning")
        #     return redirect(request.url)
        # ------------------------------------------------

        top_probs, top_indices = prediction(file_path)

        results = []
        for i in range(len(top_probs)):
            idx = top_indices[i]
            disease_record = disease_info.loc[idx]
            supplement_record = supplement_info.loc[idx]

            result = {
                'disease_name': disease_record['disease_name_es'],
                'probability': round(top_probs[i] * 100, 2),
                'description': disease_record['description_es'],
                'prevent': disease_record['steps_es'],
                'supplement_name': supplement_record['supplement_name_es'],
                'supplement_image': supplement_record['supplement_image'],
                'buy_link': supplement_record['buy_link']
            }
            results.append(result)

        # --- GUARDAR EN BASE DE DATOS (HU-14) ---
        if current_user.is_authenticated:
            # Guardamos el resultado principal (top 1)
            top_result = results[0]
            new_diagnosis = Diagnosis(
                user_id=current_user.id,
                image_path=f"uploads/{filename}",
                disease_name=top_result['disease_name'],
                probability=top_result['probability']
            )
            db.session.add(new_diagnosis)
            db.session.commit()

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
