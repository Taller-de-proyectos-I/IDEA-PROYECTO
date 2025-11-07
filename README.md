# Implementación de sistema de visión por computadora con CNN para diagnóstico de enfermedades en hojas de cultivos andinos en comunidades de Cusco
![Texto alternativo](https://img.freepik.com/premium-photo/robot-hand-holding-small-plants-with-environment-icon_296091-7706.jpg)

<br>
## Tablero Kanban
<br>

[Sprint 1](https://app.asana.com/1/1110263497108613/project/1211417566709710/board/1211417623865614)
<br>

[Sprint 2](https://app.asana.com/1/1110263497108613/project/1211417594384707/list/1211417654109355)
<br>

[Sprint 3](https://app.asana.com/1/1110263497108613/project/1211488064071969/list/1211488414723182)
<br>

[Sprint 4](https://app.asana.com/1/1110263497108613/project/1211593957703665/list/1211593881579449)
<br>

[Sprint 5](https://app.asana.com/1/1110263497108613/project/1211644724450729/list/1211644604853759)
<br>

[Paper](https://drive.google.com/drive/folders/19r0NLbhYGRLrOVhUdpl79FP8IHCLgemc?usp=sharing)


# PlantAndes - Documentación Técnica

## 1. Arquitectura General del Sistema

PlantAndes es una aplicación web monolítica construida con el microframework Flask en Python. La arquitectura está diseñada para ser modular y escalable, separando las responsabilidades en componentes lógicos.

![Diagrama de Arquitectura de PlantAndes](https://www.sngular.com/images/1/1398/original/arquitectura-monolitica-vs-arquitectura-microservicios.webp)

*   **Frontend (Cliente):**
    *   **Tecnologías:** HTML5, CSS3, JavaScript (ES6).
    *   **Renderizado:** La interfaz se renderiza principalmente en el servidor utilizando el motor de plantillas **Jinja2**, integrado en Flask. Esto simplifica el estado del frontend y lo hace ligero.
    *   **Interactividad:** Se utiliza JavaScript "vanilla" para la interactividad del lado del cliente, como menús desplegables, validaciones de formularios, y comunicación asíncrona (AJAX/Fetch) con el backend para funciones como el chat con la IA y el guardado de notas.

*   **Backend (Servidor):**
    *   **Framework:** **Flask (Python)**. Gestiona el enrutamiento, la lógica de negocio, la autenticación de usuarios y la interacción con la base deatos.
    *   **Base de Datos:** **SQLite** a través del ORM **SQLAlchemy**. Se eligió SQLite por su simplicidad y facilidad de configuración para desarrollo y despliegues de pequeña a mediana escala. Las migraciones de la base de datos se gestionan con **Flask-Migrate**.
    *   **Autenticación:** Se maneja con **Flask-Login** para gestionar las sesiones de usuario, proteger rutas y controlar el acceso basado en roles (usuario y administrador).
    *   **Internacionalización (i18n):** Se utiliza **Flask-Babel** para soportar múltiples idiomas (Español y Quechua), permitiendo la traducción de todos los textos de la interfaz y mensajes del sistema.

*   **Modelo de Inteligencia Artificial:**
    *   **Framework:** **PyTorch**.
    *   **Arquitectura:** Se utiliza un modelo de **Red Neuronal Convolucional (CNN)** personalizado, entrenado para clasificar imágenes de hojas de plantas en 38 clases de enfermedades distintas y una clase saludable. El modelo está optimizado para ejecutarse en CPU, garantizando su funcionamiento en entornos de servidor sin GPU.
    *   **Asistente IA Generativa:** Se integra con la API de **DeepSeek** (compatible con la API de OpenAI) para proporcionar un asistente de chat contextual que ayuda a los usuarios a entender los diagnósticos.

---

## 2. Flujo de Datos

### Flujo de Diagnóstico Principal

1.  **Carga de Imagen:** El usuario selecciona una o varias imágenes desde el formulario en la página "Motor IA".
2.  **Recepción en Backend:** La ruta `/index` de Flask recibe las imágenes.
3.  **Validación:** Cada imagen pasa por una validación de calidad (`check_blur_from_stream`) para descartar fotos borrosas que podrían dar un mal diagnóstico.
4.  **Predicción:** La imagen se pasa a la función `prediction()`, que la pre-procesa (redimensiona a 224x224, convierte a tensor) y la alimenta al modelo CNN de PyTorch.
5.  **Resultados del Modelo:** El modelo devuelve un vector de probabilidades para las 39 clases. Se seleccionan las 3 más probables.
6.  **Enriquecimiento de Datos:** Los índices de las enfermedades predichas se usan para buscar información detallada (nombre en español, descripción, tratamiento) en los DataFrames de `pandas` cargados en memoria.
7.  **Guardado (si está autenticado):** El diagnóstico principal se guarda en la tabla `Diagnosis` de la base de datos, asociado al ID del usuario.
8.  **Renderizado:** Los resultados enriquecidos se envían a la plantilla `index.html`, que los muestra al usuario.

### Flujo de Autenticación

1.  **Registro:** El usuario completa el formulario de registro. La ruta `/register` valida que el email y el nombre de usuario no existan, hashea la contraseña y crea un nuevo registro en la tabla `User`.
2.  **Inicio de Sesión:** El usuario envía su email y contraseña. La ruta `/login` busca al usuario por email y verifica la contraseña usando el método `check_password()`. Si es exitoso, Flask-Login crea una sesión segura para el usuario.

---

## 3. Dependencias e Instalación

Para ejecutar el proyecto localmente, sigue estos pasos:

1.  **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/Taller-de-proyectos-I/plantandes.git
    cd "Flask Deployed App"
    ```

2.  **Crear un Entorno Virtual:**
    ```bash
    # En Windows
    python -m venv venv
    venv\Scripts\activate

    # En macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar Dependencias:**
    Asegúrate de tener un archivo `requirements.txt` con todas las librerías. Luego, ejecuta:
    ```bash
    pip install -r requirements.txt
    ```
    Las dependencias clave incluyen: `Flask`, `torch`, `torchvision`, `pandas`, `Pillow`, `SQLAlchemy`, `Flask-SQLAlchemy`, `Flask-Migrate`, `Flask-Login`, `Flask-Babel`, `openai`, `requests`, `deep-translator`.

4.  **Configurar Variables de Entorno:**
    Crea un archivo `.env` en la raíz del proyecto y añade las siguientes claves (no subas este archivo a Git):
    ```
    SECRET_KEY='una-clave-muy-secreta-y-larga'
    MAIL_USERNAME='tu-correo@gmail.com'
    MAIL_PASSWORD='tu-contraseña-de-aplicacion-de-gmail'
    DEEPSEEK_API_KEY='sk-...'
    ```

5.  **Inicializar y Migrar la Base de Datos:**
    Ejecuta los siguientes comandos en orden:
    ```bash
    flask db init  # Solo la primera vez
    flask db migrate -m "Initial migration"
    flask db upgrade
    ```

6.  **Poblar Datos de Ubicación:**
    Este comando obtiene todos los distritos de Perú desde una API y los guarda en la base de datos.
    ```bash
    flask seed-zones
    ```

7.  **Ejecutar la Aplicación:**
    ```bash
    python app.py
    ```
    La aplicación estará disponible en `http://127.0.0.1:5000`.

---

## 4. Anexos: Resultados del Modelo

El modelo CNN fue entrenado y validado, mostrando un rendimiento robusto para la tarea de clasificación.

*   **Precisión (Accuracy):** El modelo alcanzó una precisión de validación superior al **90%**, lo que indica una alta fiabilidad en sus diagnósticos.

*   **Curva de Pérdida (Loss Curve):** La gráfica de pérdida muestra una convergencia clara entre la pérdida de entrenamiento y la de validación, lo que sugiere que el modelo ha generalizado bien y no sufre de un sobreajuste (overfitting) significativo.

    !Curva de Pérdida

*   **Matriz de Confusión:** La matriz de confusión visualiza el rendimiento por clase. La diagonal principal está fuertemente marcada, indicando un alto número de aciertos para cada enfermedad y muy pocas confusiones entre clases distintas.

    !Matriz de Confusión
