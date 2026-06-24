# ==============================================================================
# PlantAndes - Detección de Enfermedades en Cultivos Andinos
#
# Copyright (c) 2025 JherzonDev. Todos los derechos reservados.
#
# Autor: JherzonDev
#
# Este software se proporciona "tal cual", sin garantía de ningún tipo.
# ==============================================================================

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
import io, csv
from deep_translator import GoogleTranslator
from datetime import datetime, time, timedelta
from openai import OpenAI
import requests
from collections import Counter
import pytz
# --- IMPORTACIONES DE MODELOS Y AUTENTICACIÓN ---
from models import db, User, Zone, Diagnosis, Notification, FarmTask, ChatMessage
from flask_migrate import Migrate
from image_validator import check_blur_from_stream
from sqlalchemy import func, event
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

# --- Formato encoder para las calumnas y contenido que contenga caracteres especiales ---
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
# Aumentar el timeout para evitar errores de "database is locked" en SQLite
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'connect_args': {'timeout': 20}
}

# --- CONFIGURACIÓN DE INTERNACIONALIZACIÓN (i18n) ---
app.config['LANGUAGES'] = {
    'es': 'Español',
    'qu': 'Quechua'
}
app.config['BABEL_DEFAULT_LOCALE'] = 'es'

# --- INICIALIZAR EXTENSIONES ---
db.init_app(app)
migrate = Migrate(app, db)
with app.app_context():
    db.create_all()
from flask_babel import Babel, gettext

babel = Babel()
def get_locale():
    # 1. Usar el idioma de la sesión si está disponible
    if 'language' in session and session['language'] in app.config['LANGUAGES']:
        return session['language']
    # 2. De lo contrario, usar el idioma preferido del navegador
    return request.accept_languages.best_match(app.config['LANGUAGES'].keys())

babel.init_app(app, locale_selector=get_locale)

# --- CONFIGURACIÓN DE ZONA HORARIA PARA SQLITE ---
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Activa las claves foráneas en SQLite al establecer una conexión."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

login_manager = LoginManager()
login_manager.init_app(app)
# Redirige a la página de login si se intenta acceder a una ruta protegida
login_manager.login_view = 'login' 
login_manager.login_message = gettext("Por favor, inicie sesión para acceder a esta página.")
login_manager.login_message_category = "info"

@app.template_filter('localtime')
def localtime_filter(utc_dt):
    """Filtro de Jinja2 para convertir una fecha UTC a la hora local de Perú."""
    if not utc_dt:
        return ""
    local_tz = pytz.timezone('America/Lima')
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_dt.strftime('%d/%m/%Y %I:%M %p')

@app.route('/language/<lang>')
def set_language(lang):
    if lang in app.config['LANGUAGES']:
        session['language'] = lang
    # Redirigir a la página anterior o a la página principal
    return redirect(request.referrer or url_for('home_page'))

@app.context_processor
def inject_notifications():
    """Inyecta las notificaciones no leídas en todas las plantillas para el usuario actual."""
    if current_user.is_authenticated:
        unread_notifications = Notification.query.filter_by(user_id=current_user.id, is_read=False).order_by(Notification.created_at.desc()).all()
        return dict(unread_notifications=unread_notifications)
    return dict(unread_notifications=[])

@app.route('/notifications/read/<int:notification_id>')
@login_required
def mark_notification_as_read(notification_id):
    """Marca una notificación como leída y redirige al enlace asociado."""
    notification = Notification.query.get_or_404(notification_id)
    if notification.user_id == current_user.id:
        notification.is_read = True
        db.session.commit()
    return redirect(notification.link or url_for('home_page'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Clave secreta para mensajes flash \ entorno local 
app.secret_key = os.environ.get('SECRET_KEY', 'una-clave-secreta-por-defecto-para-desarrollo')

# --- Configuración de la API de IA Generativa (DeepSeek) ---
app.config['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY', '')

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
    # Limpiar contadores de la sesión al cerrar sesión
    session.pop('analysis_count', None)
    session.pop('feedback_shown_in_session', None)
    return redirect(url_for('home_page'))

# --- RUTA PARA EL ASISTENTE DE IA GENERATIVA ---
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """Maneja las peticiones al chat contextual de diagnóstico."""
    data = request.json
    user_message = data.get('message')
    context = data.get('context') 
    language = data.get('language', 'Español')
    diagnosis_id = data.get('diagnosis_id')

    if not user_message or not context:
        return jsonify({'error': 'Falta mensaje o contexto'}), 400

    # Guardar mensaje del usuario si hay diagnosis_id
    if diagnosis_id:
        try:
            user_msg = ChatMessage(diagnosis_id=diagnosis_id, sender='user', message=user_message)
            db.session.add(user_msg)
            db.session.commit()
        except Exception as e:
            print(f"Error guardando mensaje de usuario: {e}")

    # Prompt de sistema: Define la "personalidad" y el rol del asistente de IA.
    system_prompt = (
        "Eres 'AndesGPT', un asistente experto en agronomía y fitopatología, especializado en cultivos andinos. "
        "Tu propósito es ayudar a los agricultores a entender y manejar las enfermedades de sus plantas. "
        "Actúa de forma amigable, profesional y da respuestas claras, estructuradas y accionables. "
        "REGLAS DE FORMATO:\n"
        "- Divide la respuesta en párrafos claros separados por una línea en blanco (doble salto de línea).\n"
        "- Utiliza títulos descriptivos en negrita (ej: **• Síntomas Principales:**, **• Tratamiento Recomendado:**).\n"
        "- Usa listas numeradas o con viñetas claras para pasos o alternativas.\n"
        "- Agrega emojis oportunos y contextuales (ej: 🌿, 🐛, 🧪, 💧, ⚠️) para hacer la lectura más amigable y estructurada, sin saturar.\n"
        f"Utiliza el siguiente contexto para responder la pregunta del usuario. No inventes información. Responde siempre en el idioma: {language}."
    )

    # Prompt de usuario: Combina el contexto del diagnóstico con la pregunta específica del usuario.
    full_prompt = (
        f"Contexto del diagnóstico:\n"
        f"- Enfermedad detectada: {context.get('disease_name')}\n"
        f"- Probabilidad de diagnóstico: {context.get('probability')}%\n"
        f"- Descripción: {context.get('description')}\n"
        f"- Recomendaciones: {context.get('prevent')}\n\n"
        f"Pregunta del usuario: {user_message}"
    )

    def generate():
        """Función generadora para la respuesta en streaming y almacenamiento posterior."""
        full_reply = ""
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
                    full_reply += content
                    yield content
            
            # Guardar la respuesta completa de la IA una vez finalizado el stream
            if diagnosis_id and full_reply:
                with app.app_context():
                    try:
                        ai_msg = ChatMessage(diagnosis_id=diagnosis_id, sender='ai', message=full_reply)
                        db.session.add(ai_msg)
                        db.session.commit()
                    except Exception as db_err:
                        db.session.rollback()
                        print(f"Error guardando respuesta de IA en base de datos: {db_err}")
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

    system_prompt = ( # Personalidad del asistente para el chat general.
        "Eres 'AndesGPT', un asistente experto en agronomía y fitopatología, especializado en cultivos andinos. "
        "Tu propósito es ayudar a los agricultores. Si te preguntan sobre una enfermedad específica, anímales a usar el 'Motor IA' "
        "y subir una foto para un diagnóstico preciso. Puedes responder preguntas generales sobre agricultura, "
        "técnicas de cultivo, plagas comunes, etc. "
        "REGLAS DE FORMATO:\n"
        "- Divide la respuesta en párrafos claros separados por una línea en blanco (doble salto de línea).\n"
        "- Utiliza títulos descriptivos en negrita (ej: **• Consejo Agrícola:**, **• Prevención:**).\n"
        "- Usa listas numeradas o con viñetas claras para enumerar pasos, especies o consejos.\n"
        "- Agrega emojis oportunos y contextuales (ej: 🌿, 🚜, 🐛, 💧, ☀️) para hacer la lectura más amigable y estructurada, sin saturar.\n"
        f"Responde siempre en el idioma: {language}."
    )

    def generate():
        """Función generadora para enviar la respuesta de la IA en trozos (streaming)."""
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

@app.route('/api/clear_chat/<int:diagnosis_id>', methods=['POST'])
@login_required
def clear_chat_api(diagnosis_id):
    """Limpia todo el historial de chat de un diagnóstico específico."""
    diag = Diagnosis.query.filter_by(id=diagnosis_id, user_id=current_user.id).first()
    if not diag:
        return jsonify({'success': False, 'error': 'Diagnóstico no encontrado o no autorizado'}), 404
    
    try:
        ChatMessage.query.filter_by(diagnosis_id=diagnosis_id).delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error borrando chat de BD: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.cli.command("seed-zones")
def seed_zones():
    """Poblar la base de datos con todos los distritos de Perú desde una API externa."""
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

        for dep_name, provincias in ubigeos.items():
            if not isinstance(provincias, dict):
                continue

            for prov_name, distritos in provincias.items():
                if not isinstance(distritos, dict):
                    continue

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

def get_mock_weather(zone):
    # Genera clima simulado de forma estable basada en el ubigeo
    h = hash(zone.district_name or "Cusco") % 5
    temp_min = 8 + (h % 3)
    temp_max = 18 + (h % 4)
    temp = (temp_min + temp_max) // 2
    humidity = 60 + (h * 7)
    conditions = ["Soleado", "Parcialmente Nublado", "Lluvia Ligera", "Nublado", "Llovizna"]
    condition = conditions[h % len(conditions)]
    
    mock_icons = ["01d", "02d", "09d", "04d", "10d"]
    icon = mock_icons[h % len(mock_icons)]
    
    advisory = []
    if humidity > 80:
        advisory.append("Riesgo elevado de Tizón Tardío en Papa y Mancha Foliar en Tomate debido a la alta humedad. Considere tratamientos preventivos.")
    if temp < 10:
        advisory.append("Temperaturas bajas registradas. Proteja los brotes jóvenes contra heladas nocturnas.")
    if condition in ["Lluvia Ligera", "Llovizna"]:
        advisory.append("Se registran precipitaciones. Evite aplicar fungicidas de contacto o fertilizantes líquidos expuestos.")
    else:
        advisory.append("Buen clima para labores de deshierbe y fertilización de cobertura.")
        
    return {
        'temp': temp,
        'temp_min': temp_min,
        'temp_max': temp_max,
        'humidity': humidity,
        'condition': condition,
        'icon': icon,
        'advisories': advisory
    }

def get_real_weather(zone):
    api_key = "a20c9a2eeb4ae79f8d6e46253db2c13d"
    # Buscar por diferentes combinaciones geográficas en Perú progresivamente
    queries = [
        f"{zone.district_name},{zone.province_name},PE",
        f"{zone.province_name},PE",
        f"{zone.department_name},PE",
        "Cusco,PE"
    ]
    for q in queries:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={q}&appid={api_key}&units=metric&lang=es"
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                data = r.json()
                temp = round(data['main']['temp'])
                temp_min = round(data['main']['temp_min'])
                temp_max = round(data['main']['temp_max'])
                humidity = data['main']['humidity']
                
                # Mapeo de descripción para español más natural y profesional
                raw_condition = data['weather'][0]['description'].lower()
                condition_map = {
                    'cielo claro': 'Cielo Despejado',
                    'algo de nubes': 'Pocas Nubes',
                    'nubes dispersas': 'Parcialmente Nublado',
                    'nubes': 'Nublado',
                    'muy nuboso': 'Nublado',
                    'cubierto': 'Totalmente Nublado',
                    'lluvia ligera': 'Lluvia Ligera',
                    'lluvia moderada': 'Lluvia Moderada',
                    'lluvia fuerte': 'Lluvia Fuerte',
                    'llovizna': 'Llovizna',
                    'llovizna ligera': 'Llovizna Ligera',
                    'tormenta': 'Tormenta',
                    'nieve': 'Nieve',
                    'niebla': 'Niebla',
                    'neblina': 'Neblina',
                }
                condition = condition_map.get(raw_condition, raw_condition.capitalize())
                icon = data['weather'][0]['icon']
                
                advisory = []
                if humidity > 80:
                    advisory.append("Humedad muy alta detectada en tu zona. Hay un riesgo elevado de Tizón Tardío en Papa y Mancha Foliar en Tomate. Evite regar por aspersión y aplique preventivos si es necesario.")
                if temp < 10:
                    advisory.append("Temperaturas bajas. Existe riesgo de heladas foliares que pueden dañar los brotes tiernos. Proteja cultivos vulnerables.")
                if 'lluvia' in condition.lower() or 'llovizna' in condition.lower():
                    advisory.append("Precipitaciones en curso. Evite aplicar fungicidas de contacto o fertilizantes líquidos ya que se lavarán con el agua de lluvia.")
                else:
                    advisory.append("Buen clima para labores generales, deshierbe y fertilización dirigida.")
                    
                return {
                    'temp': temp,
                    'temp_min': temp_min,
                    'temp_max': temp_max,
                    'humidity': humidity,
                    'condition': condition,
                    'icon': icon,
                    'advisories': advisory,
                    'is_real': True,
                    'queried': q
                }
        except Exception as e:
            print(f"Error cargando clima para {q}: {e}")
            
    # Si falla la API, retornar clima simulado como fallback
    return get_mock_weather(zone)

@app.route('/')
def home_page():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Panel central del agricultor registrado."""
    # 1. Tareas pendientes del día o próximas
    tasks = current_user.tasks.filter_by(status='pending').order_by(FarmTask.task_date.asc()).limit(5).all()
    
    # 2. Diagnósticos recientes
    recent_diagnoses = current_user.diagnoses.order_by(Diagnosis.created_at.desc()).limit(3).all()
    
    # 3. Alertas de brote en la misma zona
    ALERT_THRESHOLD = 3
    TIME_WINDOW_DAYS = 14
    peru_tz = pytz.timezone('America/Lima')
    start_date = datetime.now(peru_tz) - timedelta(days=TIME_WINDOW_DAYS)
    
    outbreaks = db.session.query(
        Diagnosis.disease_name,
        func.count(Diagnosis.id).label('case_count')
    ).join(User, User.id == Diagnosis.user_id)\
     .filter(User.zone_id == current_user.zone_id)\
     .filter(Diagnosis.created_at >= start_date)\
     .group_by(Diagnosis.disease_name)\
     .having(func.count(Diagnosis.id) >= ALERT_THRESHOLD)\
     .all()
     
    alerts = [{'disease': o.disease_name, 'count': o.case_count} for o in outbreaks]
    
    return render_template('dashboard.html', 
                           tasks=tasks, 
                           diagnoses=recent_diagnoses, 
                           alerts=alerts)

@app.route('/api/weather')
@login_required
def api_weather():
    """Obtiene el clima en tiempo real de forma asíncrona para evitar bloquear la carga de la página."""
    zone = current_user.zone
    if not zone:
        return jsonify({'error': 'Ubicación no configurada.'}), 400
    weather = get_real_weather(zone)
    return jsonify(weather)

@app.route('/tasks', methods=['GET', 'POST'])
@login_required
def tasks_page():
    """Gestión de tareas de cultivo del agricultor."""
    if request.method == 'POST':
        title = request.form.get('title')
        crop_type = request.form.get('crop_type')
        description = request.form.get('description')
        task_date_str = request.form.get('task_date')
        
        if not title or not crop_type or not task_date_str:
            flash(gettext('Faltan datos requeridos.'), 'danger')
            return redirect(url_for('tasks_page'))
            
        try:
            task_date = datetime.strptime(task_date_str, '%Y-%m-%d').date()
            new_task = FarmTask(
                user_id=current_user.id,
                title=title,
                crop_type=crop_type,
                description=description,
                task_date=task_date
            )
            db.session.add(new_task)
            db.session.commit()
            
            # Notificación en la app
            notif = Notification(
                user_id=current_user.id,
                message=f"Nueva tarea '{title}' agendada para el {task_date.strftime('%d/%m/%Y')}.",
                link=url_for('tasks_page')
            )
            db.session.add(notif)
            db.session.commit()
            
            flash(gettext('Tarea agregada exitosamente.'), 'success')
        except ValueError:
            flash(gettext('Formato de fecha inválido.'), 'danger')
            
        return redirect(url_for('tasks_page'))
        
    # GET: Listar todas las tareas del usuario agrupadas/ordenadas
    tasks = current_user.tasks.order_by(FarmTask.task_date.asc(), FarmTask.status.desc()).all()
    return render_template('tasks.html', tasks=tasks)

@app.route('/tasks/toggle/<int:task_id>', methods=['POST'])
@login_required
def toggle_task(task_id):
    """Marca una tarea como completada o pendiente."""
    task = FarmTask.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Acceso no autorizado'}), 403
        
    task.status = 'completed' if task.status == 'pending' else 'pending'
    db.session.commit()
    
    # Crear notificación dinámica
    state_msg = 'completada' if task.status == 'completed' else 'marcada como pendiente'
    notif = Notification(
        user_id=current_user.id,
        message=f"Tarea '{task.title}' {state_msg}.",
        link=url_for('tasks_page')
    )
    db.session.add(notif)
    db.session.commit()
    
    return jsonify({'success': True, 'status': task.status})

@app.route('/tasks/delete/<int:task_id>', methods=['POST'])
@login_required
def delete_task(task_id):
    """Elimina una tarea del diario."""
    task = FarmTask.query.get_or_404(task_id)
    if task.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Acceso no autorizado'}), 403
        
    db.session.delete(task)
    db.session.commit()
    return jsonify({'success': True})

# --- RUTAS DE AUTENTICACIÓN ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    if request.method == 'POST':
        username = request.form.get('username')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        zone_id = request.form.get('district')
        address = request.form.get('address')

        # Validaciones
        if User.query.filter_by(username=username).first():
            flash(gettext('El nombre de usuario ya existe.'), 'warning')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash(gettext('El correo electrónico ya está registrado.'), 'warning')
            return redirect(url_for('register'))

        new_user = User(
            username=username,
            first_name=first_name,
            last_name=last_name,
            email=email,
            zone_id=zone_id,
            address=address)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash(gettext('¡Registro exitoso! Ahora puedes iniciar sesión.'), 'success')
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
            # Reiniciar el contador de análisis al iniciar sesión
            session.pop('feedback_shown_in_session', None)
            session['analysis_count'] = 0
            return redirect(url_for('home_page'))
        else:
            flash(gettext('Credenciales inválidas. Por favor, inténtalo de nuevo.'), 'danger')

    return render_template('login.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message_body = request.form['message']

        subject = f"Nuevo Mensaje de Contacto de: {name}"
        try:
            msg = Message(
                subject=subject,
                recipients=['71827961@continental.edu.pe'],
                reply_to=email
            )
            msg.html = render_template('email_template.html', name=name, email=email, message_body=message_body)
            mail.send(msg)
            flash(gettext('¡Gracias por tu mensaje! Nos pondremos en contacto contigo pronto.'), 'success')
        except Exception as e:
            print(f"Error al enviar correo de contacto: {e}")
            flash(gettext('Tu mensaje fue recibido, pero no se pudo enviar el correo de confirmación en este momento.'), 'warning')
        return redirect(url_for('contact'))
    return render_template('contact-us.html')

@app.route('/index', methods=['GET', 'POST'])
def ai_engine_page():
    """Página principal del motor de IA para el diagnóstico de enfermedades."""
    if request.method == 'POST':
        show_feedback_form = False
        images = request.files.getlist('image')
        if not images or all(img.filename == '' for img in images):
            # Si no se sube ninguna imagen, se muestra un error.
            flash(gettext("No se seleccionó ninguna imagen."), "error")
            return redirect(request.url)

        all_top_predictions = []
        first_image_path = None
        new_diagnosis_id = None
        quality_scores = []

        # 1. Procesar cada imagen subida.
        for image in images:
            image_stream = image.read()

            # --- Validación de Calidad ---
            quality_score, is_blurry = check_blur_from_stream(image_stream, threshold=100.0)
            if is_blurry:
                flash(gettext("La imagen parece borrosa (Puntuación de nitidez: %(score).2f). Para un mejor resultado, sube una imagen más nítida.", score=quality_score), "danger")
                return redirect(request.url)

            quality_scores.append(quality_score)

            # Guardar la imagen temporalmente para la predicción.
            filename = secure_filename(image.filename)
            file_path = os.path.join('static/uploads', filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(image_stream)

            if not first_image_path:
                first_image_path = f"uploads/{filename}"
            # Realizar la predicción con el modelo CNN.
            top_probs, top_indices = prediction(file_path)

            # Guardar solo la predicción principal (top 1) de cada imagen.
            top_disease_idx = top_indices[0]
            top_disease_name = disease_info.loc[top_disease_idx]['disease_name_es']
            top_disease_prob = round(top_probs[0] * 100, 2)
            all_top_predictions.append({'name': top_disease_name, 'prob': top_disease_prob})

        # --- Lógica de Consolidación para subir varias imagenes ---
        consolidated_result = None
        if len(images) > 1:
            disease_counts = Counter(p['name'] for p in all_top_predictions)
            most_common_disease_name = disease_counts.most_common(1)[0][0]
            probs_for_common_disease = [p['prob'] for p in all_top_predictions if p['name'] == most_common_disease_name]
            average_prob = round(sum(probs_for_common_disease) / len(probs_for_common_disease), 2)
            consolidated_result = {
                'disease_name': most_common_disease_name,
                'probability': average_prob
            }

        # 3. Guardar diagnóstico y gestionar la lógica de feedback para usuarios autenticados.
        if current_user.is_authenticated:
            # Se usa la sesión para contar cuántos análisis ha hecho el usuario.
            current_analysis_count = session.get('analysis_count', 0)

            # Para no molestar al usuario, el feedback se pide solo una vez por sesión.
            feedback_already_shown = session.get('feedback_shown_in_session', False)

            # El modal de feedback se muestra en el segundo análisis de la sesión.
            show_feedback_form = (current_analysis_count == 1 and not feedback_already_shown)

            # Si se muestra el modal, se marca para no volver a mostrarlo.
            if show_feedback_form:
                session['feedback_shown_in_session'] = True

            # Incrementar el contador para el próximo análisis.
            session['analysis_count'] = current_analysis_count + 1

            final_disease_name = consolidated_result['disease_name'] if consolidated_result else all_top_predictions[0]['name']
            final_probability = consolidated_result['probability'] if consolidated_result else all_top_predictions[0]['prob']
            avg_quality_score = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0

            new_diagnosis = Diagnosis(
                user_id=current_user.id,
                image_path=first_image_path,
                disease_name=final_disease_name,
                probability=final_probability,
                image_quality_score=avg_quality_score
            )
            db.session.add(new_diagnosis)
            db.session.commit()
            db.session.refresh(new_diagnosis)
            new_diagnosis_id = new_diagnosis.id

            # Después de guardar, se verifica si este nuevo caso genera una alerta de brote.
            check_and_generate_alerts(new_diagnosis)
            
            # Redirigir al mismo endpoint usando GET para cargar los resultados y la vista estilo chat
            show_feedback_arg = 1 if show_feedback_form else 0
            return redirect(url_for('ai_engine_page', diagnosis_id=new_diagnosis_id, show_feedback=show_feedback_arg))

        # 4. Preparar los resultados detallados para mostrar en la plantilla (si no está autenticado).
        # Se usan las predicciones de la *primera* imagen para mostrar el top 3.
        first_image_prediction = prediction(os.path.join('static', first_image_path))
        display_results = []
        for i in range(len(first_image_prediction[0])):
            idx = first_image_prediction[1][i]
            prob = round(first_image_prediction[0][i] * 100, 2)
            disease_record = disease_info.loc[idx]
            supplement_record = supplement_info.loc[idx]
            display_results.append({
                'disease_name': disease_record['disease_name_es'],
                'probability': prob,
                'description': disease_record['description_es'],
                'prevent': disease_record['steps_es'],
                'supplement_name': supplement_record['supplement_name_es'],
                'supplement_image': supplement_record['supplement_image'],
                'buy_link': supplement_record['buy_link']
            })

        return render_template('index.html',
                               results=display_results,
                               user_image=first_image_path,
                               new_diagnosis_id=None,
                               consolidated_result=consolidated_result,
                               image_count=len(images),
                               show_feedback_form=False)

    # Para peticiones GET
    previous_diagnoses = []
    selected_diagnosis = None
    display_results = None
    user_image = None
    chat_messages = []
    show_feedback = request.args.get('show_feedback') == '1'
    diag_id = request.args.get('diagnosis_id')

    if current_user.is_authenticated:
        previous_diagnoses = Diagnosis.query.filter_by(user_id=current_user.id).order_by(Diagnosis.created_at.desc()).all()
        if diag_id:
            selected_diagnosis = Diagnosis.query.filter_by(id=diag_id, user_id=current_user.id).first()
            if selected_diagnosis:
                user_image = selected_diagnosis.image_path
                matched_rows = disease_info[disease_info['disease_name_es'] == selected_diagnosis.disease_name]
                if not matched_rows.empty:
                    idx = matched_rows.index[0]
                    disease_record = disease_info.loc[idx]
                    supplement_record = supplement_info.loc[idx]
                    display_results = [{
                        'disease_name': disease_record['disease_name_es'],
                        'probability': selected_diagnosis.probability,
                        'description': disease_record['description_es'],
                        'prevent': disease_record['steps_es'],
                        'supplement_name': supplement_record['supplement_name_es'],
                        'supplement_image': supplement_record['supplement_image'],
                        'buy_link': supplement_record['buy_link']
                    }]
                chat_messages = ChatMessage.query.filter_by(diagnosis_id=selected_diagnosis.id).order_by(ChatMessage.created_at.asc()).all()

    return render_template('index.html',
                           results=display_results,
                           user_image=user_image,
                           new_diagnosis_id=selected_diagnosis.id if selected_diagnosis else None,
                           previous_diagnoses=previous_diagnoses,
                           chat_messages=chat_messages,
                           show_feedback_form=show_feedback)

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Permite a un usuario editar su perfil y al administrador gestionar otros usuarios."""
    users = None
    if current_user.role == 'admin':
        # admin puede ver a todos los demás usuarios
        users = User.query.filter(User.id != current_user.id).order_by(User.username).all()

    if request.method == 'POST':
        # Actualizar nombres y apellidos
        current_user.first_name = request.form.get('first_name')
        current_user.last_name = request.form.get('last_name')

        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # Lógica para cambiar la contraseña (solo si se proporcionan nuevos valores).
        if new_password:
            if new_password != confirm_password:
                flash(gettext('Las nuevas contraseñas no coinciden. Por favor, inténtalo de nuevo.'), 'danger')
                return redirect(url_for('edit_profile'))

            current_user.set_password(new_password)
            flash(gettext('Tu contraseña ha sido actualizada con éxito.'), 'success')

        db.session.commit()
        flash(gettext('Tu perfil ha sido actualizado con éxito.'), 'success')
        return redirect(url_for('edit_profile'))

    return render_template('edit_profile.html', users=users)

@app.route('/admin/update_role/<int:user_id>', methods=['POST'])
@login_required
def update_user_role(user_id):
    """Ruta para que un administrador cambie el rol de otro usuario."""
    if current_user.role != 'admin':
        flash(gettext('No tienes permiso para realizar esta acción.'), 'danger')
        return redirect(url_for('home_page'))

    user_to_update = User.query.get_or_404(user_id)
    user_to_update.role = request.form.get('role')
    db.session.commit()
    flash(f"El rol de {user_to_update.username} ha sido actualizado a '{user_to_update.role}'.", 'success')
    return redirect(url_for('edit_profile'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    # TODO: Implementar la lógica de eliminación de usuario
    flash(gettext("Funcionalidad de eliminar usuario (ID: %(user_id)s) pendiente de implementación.", user_id=user_id), 'info')
    return redirect(url_for('edit_profile'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    """Muestra el panel de administración con estadísticas de la plataforma."""
    if current_user.role != 'admin':
        flash(gettext('Acceso denegado. Esta sección es solo para administradores.'), 'danger')
        return redirect(url_for('home_page'))

    # --- Estadísticas Clave ---
    total_users = User.query.count()
    total_diagnoses = Diagnosis.query.count()

    # Nuevos usuarios en los últimos 7 días.
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    new_users_count = User.query.filter(User.created_at >= seven_days_ago).count()

    # Diagnósticos realizados hoy, considerando la zona horaria de Perú.
    today_start = datetime.combine(datetime.now(pytz.timezone('America/Lima')).date(), time.min)
    diagnoses_today_count = Diagnosis.query.filter(Diagnosis.created_at >= today_start).count()

    # --- Top 5s ---
    # Usuarios más activos (con más diagnósticos)
    most_active_users = db.session.query(
        User.username, func.count(Diagnosis.id).label('diag_count')
    ).join(Diagnosis).group_by(User.id).order_by(func.count(Diagnosis.id).desc()).limit(5).all()

    # Enfermedades más comunes
    most_common_diseases = db.session.query(
        Diagnosis.disease_name, func.count(Diagnosis.id).label('disease_count')
    ).group_by(Diagnosis.disease_name).order_by(func.count(Diagnosis.id).desc()).limit(5).all()

    # --- Preparar datos para los gráficos ---
    disease_labels = [d.disease_name for d in most_common_diseases]
    disease_data = [d.disease_count for d in most_common_diseases]

    user_labels = [u.username for u in most_active_users]
    user_data = [u.diag_count for u in most_active_users]

    return render_template('admin_dashboard.html',
                           total_users=total_users, total_diagnoses=total_diagnoses,
                           new_users_count=new_users_count, diagnoses_today_count=diagnoses_today_count,
                           most_active_users=most_active_users, most_common_diseases=most_common_diseases,
                           disease_labels=disease_labels, disease_data=disease_data,
                           user_labels=user_labels, user_data=user_data)

@app.route('/admin/export/diagnoses')
@login_required
def export_diagnoses():
    """
    Se genera y sirve un archivo CSV con datos de diagnóstico anonimizados.
    """
    if current_user.role != 'admin':
        flash(gettext('Acceso denegado.'), 'danger')
        return redirect(url_for('home_page'))

    # 1. Realizar la consulta a la base de datos, uniendo las tablas para obtener datos de zona.
    # Se seleccionan explícitamente los campos para anonimizar los datos.
    query = db.session.query(
        Diagnosis.id,
        Diagnosis.disease_name,
        Diagnosis.probability,
        Diagnosis.created_at,
        Diagnosis.feedback_rating,
        Diagnosis.feedback_comment,
        Diagnosis.image_quality_score,
        Zone.district_name,
        Zone.province_name,
        Zone.department_name
    ).join(User, User.id == Diagnosis.user_id)\
     .join(Zone, Zone.id == User.zone_id)\
     .order_by(Diagnosis.created_at.desc()).all()

    # 2. Crear el archivo CSV en memoria
    output = io.StringIO()
    writer = csv.writer(output)

    # Escribir la fila de encabezado
    header = ['id_diagnostico', 'enfermedad', 'probabilidad', 'fecha_creacion',
              'calificacion_feedback', 'comentario_feedback', 'calidad_imagen',
              'distrito', 'provincia', 'departamento']
    writer.writerow(header)

    # Escribir las filas de datos
    for row in query:
        writer.writerow(row)

    # 3. Preparar la respuesta para la descarga
    output.seek(0)
    return Response(output, mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename=diagnosticos_anonimizados_{datetime.now().strftime('%Y-%m-%d')}.csv"})

@app.route('/update_notes', methods=['POST'])
@login_required
def update_notes():
    """Actualiza las notas de un diagnóstico existente."""
    diagnosis_id = request.form.get('diagnosis_id')
    notes = request.form.get('notes')

    diagnosis = Diagnosis.query.get_or_404(diagnosis_id)

    # Asegurarse de que el usuario solo pueda editar sus propios diagnósticos
    if diagnosis.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'No tienes permiso para editar este diagnóstico.'}), 403

    diagnosis.notes = notes
    db.session.commit()

    # Devolver una respuesta JSON para que el frontend la maneje con AJAX.
    return jsonify({'success': True, 'message': 'Nota guardada con éxito.'})

def check_and_generate_alerts(diagnosis):
    """Verifica si un nuevo diagnóstico dispara una alerta y notifica a los usuarios."""
    ALERT_THRESHOLD = 3
    TIME_WINDOW_DAYS = 14

    user = User.query.get(diagnosis.user_id)
    if not user or not user.zone_id:
        return

    # Usar la zona horaria de Perú para asegurar que la ventana de tiempo sea correcta.
    peru_tz = pytz.timezone('America/Lima')
    start_date_window = datetime.now(peru_tz) - timedelta(days=TIME_WINDOW_DAYS)

    # Contar casos de la misma enfermedad en la misma zona geográfica dentro de la ventana de tiempo.
    case_count = db.session.query(func.count(Diagnosis.id))\
        .join(User, User.id == Diagnosis.user_id)\
        .filter(User.zone_id == user.zone_id)\
        .filter(Diagnosis.disease_name == diagnosis.disease_name)\
        .filter(Diagnosis.created_at >= start_date_window).scalar()

    # Si el número de casos alcanza el umbral, verificamos si ya se envió una alerta para este brote.
    if case_count >= ALERT_THRESHOLD:
        alert_message_start = f"¡Alerta de brote! Se han detectado"
        # Evitar duplicados: buscar si ya existe una notificación para este brote.
        existing_notification = db.session.query(Notification).join(User).filter(
            User.zone_id == user.zone_id,
            Notification.message.like(f"%{diagnosis.disease_name}%"),
            Notification.created_at >= start_date_window
        )\
            .first()

        # Si no hay una notificación previa, se crea una para todos los usuarios de la zona.
        if not existing_notification:
            message = f"{alert_message_start} {case_count} casos de '{diagnosis.disease_name}' en tu zona."
            users_in_zone = User.query.filter_by(zone_id=user.zone_id).all()
            for u in users_in_zone:
                notif = Notification(user_id=u.id, message=message, link=url_for('alerts_page'))
                db.session.add(notif)
            db.session.commit()

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Guarda la retroalimentación del usuario para un diagnóstico."""
    data = request.json
    diagnosis_id = data.get('diagnosis_id')
    rating = data.get('rating')
    comment = data.get('comment')

    if not diagnosis_id or not rating:
        return jsonify({'success': False, 'message': 'Faltan datos.'}), 400

    diagnosis = Diagnosis.query.get(diagnosis_id)
    if diagnosis:
        diagnosis.feedback_rating = rating
        diagnosis.feedback_comment = comment
        db.session.commit()
        return jsonify({'success': True, 'message': '¡Gracias por tu retroalimentación!'})

    return jsonify({'success': False, 'message': 'Diagnóstico no encontrado.'}), 404

@app.route('/history')
@login_required
def history():
    """Muestra el historial de diagnósticos del usuario con opciones de filtrado."""

    # Obtener parámetros de filtro desde la URL
    disease_filter = request.args.get('disease', '')
    start_date_str = request.args.get('start_date', '')
    end_date_str = request.args.get('end_date', '')

    # Query base para los diagnósticos del usuario actual
    query = Diagnosis.query.filter_by(user_id=current_user.id)

    # Aplicar filtros si existen
    if disease_filter:
        query = query.filter(Diagnosis.disease_name.ilike(f'%{disease_filter}%'))

    if start_date_str: # Filtro por fecha de inicio.
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            query = query.filter(Diagnosis.created_at >= start_date)
        except ValueError:
            flash('Formato de fecha de inicio inválido.', 'warning')

    if end_date_str:
        # Filtro por fecha de fin. Se incluye el día completo en el rango.
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            end_date = datetime.combine(end_date, time.max)
            query = query.filter(Diagnosis.created_at <= end_date)
        except ValueError:
            flash(gettext('Formato de fecha de fin inválido.'), 'warning')

    # Ejecutar la consulta y ordenar
    diagnoses_from_db = query.order_by(Diagnosis.created_at.desc()).all()

    # Enriquecer cada diagnóstico con su descripción y recomendaciones desde el DataFrame en memoria.
    diagnoses_with_details = []
    for diag in diagnoses_from_db:
        # Buscar la información de la enfermedad en el DataFrame
        disease_details_row = disease_info[disease_info['disease_name_es'] == diag.disease_name]

        if not disease_details_row.empty:
            disease_details = disease_details_row.iloc[0]
            diagnoses_with_details.append({
                'image_path': diag.image_path,
                'disease_name': diag.disease_name,
                'probability': diag.probability,
                'created_at': diag.created_at,
                'description': disease_details['description_es'],
                'recommendations': disease_details['steps_es'],
                'notes': diag.notes
            })

    # Si es una petición AJAX (desde el filtro), devolvemos solo el fragmento HTML.
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template('_history_results.html', diagnoses=diagnoses_with_details)

    # Para la carga inicial de la página, renderizamos la página completa
    return render_template('history.html',
                           diagnoses=diagnoses_with_details,
                           disease_filter=disease_filter,
                           start_date=start_date_str,
                           end_date=end_date_str)

@app.route('/alerts')
def alerts_page():
    """Muestra una página con alertas de brotes de enfermedades por zona."""
    # 1. Definir la regla para un brote
    ALERT_THRESHOLD = 3  # A partir de 3 casos se considera alerta
    TIME_WINDOW_DAYS = 14 # En los últimos 14 días

    # Usar la zona horaria de Perú para una comparación de fechas precisa.
    peru_tz = pytz.timezone('America/Lima')
    start_date = datetime.now(peru_tz) - timedelta(days=TIME_WINDOW_DAYS)

    # 2. Consultar la base de datos para encontrar brotes
    # se agrupa por zona y enfermedad, y contamos los diagnósticos recientes
    outbreaks = db.session.query(
        Diagnosis.disease_name,
        Zone.district_name,
        Zone.province_name,
        func.count(Diagnosis.id).label('case_count')
    ).join(User, User.id == Diagnosis.user_id)\
     .join(Zone, Zone.id == User.zone_id)\
     .filter(Diagnosis.created_at >= start_date)\
     .group_by(Diagnosis.disease_name, Zone.id)\
     .having(func.count(Diagnosis.id) >= ALERT_THRESHOLD)\
     .order_by(func.count(Diagnosis.id).desc())\
     .all()

    # 3. Procesar los resultados para la plantilla, asignando un nivel de riesgo.
    alerts = []
    for outbreak in outbreaks:
        count = outbreak.case_count
        risk_level = ''
        if count >= 10:
            risk_level = 'Alto'
        elif count >= 5:
            risk_level = 'Moderado'
        else:
            risk_level = 'Bajo'

        alerts.append({
            'disease': outbreak.disease_name,
            'location': f"{outbreak.district_name}, {outbreak.province_name}",
            'count': count,
            'risk': risk_level
        })

    return render_template('alerts.html', alerts=alerts)

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    """Muestra la biblioteca de enfermedades y suplementos."""
    return render_template('market.html',
                        diseases=disease_info.to_dict(orient='records'),
                        supplements=supplement_info.to_dict(orient='records'))

@app.route('/manual')
def user_manual():
    """Muestra la página del manual de usuario."""
    return render_template('manual.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# ==============================================================================
# Fin del archivo app.py
# Autor del Script: JherzonDev
# ==============================================================================
