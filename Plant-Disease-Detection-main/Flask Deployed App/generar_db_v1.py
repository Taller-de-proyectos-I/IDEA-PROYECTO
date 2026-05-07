"""
DEFICIENCIAS IDENTIFICADAS:
  [D1] Sin claves foraneas - no hay integridad referencial
  [D2] Sin normalizacion - datos de zona duplicados en cada usuario
  [D3] Sin indices - consultas lentas en tablas grandes
  [D4] Tipos de datos incorrectos (probabilidad como TEXT en vez de REAL)
  [D5] Campos criticos sin restriccion NOT NULL
  [D6] Sin restriccion UNIQUE en email/username - permite duplicados
  [D7] Tabla monolitica "registros" mezcla diagnosticos y notificaciones
  [D8] Contrasenas almacenadas en texto plano (sin hash)
  [D9] Sin timestamp de creacion en tabla principal
  [D10] Nombres de columnas inconsistentes (mix ingles/espanol, sin convencion)
"""

import sqlite3
import os
from datetime import datetime, timedelta
import random

DB_PATH = r'C:\Users\PC\Desktop\IDEA-PROYECTO\Plant-Disease-Detection-main\Flask Deployed App\instance\plantandes_v1_old.db'

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ==============================================================================
# ESQUEMA V1 - CON DEFICIENCIAS INTENCIONADAS (para comparacion con V2)
# ==============================================================================

# [D1][D2][D5][D6][D8] - Tabla de usuarios sin normalizacion ni restricciones
cursor.execute("""
CREATE TABLE usuarios (
    id          INTEGER PRIMARY KEY,
    nombre      TEXT,
    apellido    TEXT,
    usuario     TEXT,
    correo      TEXT,
    password    TEXT,
    departamento TEXT,
    provincia   TEXT,
    distrito    TEXT,
    pais        TEXT,
    direccion   TEXT,
    tipo        TEXT
)
""")
# Deficiencias: sin UNIQUE en correo/usuario, password en texto plano,
# zona embebida directamente (no normalizada), sin created_at

# [D3][D4][D7][D9] - Tabla monolitica que mezcla diagnosticos con otros datos
cursor.execute("""
CREATE TABLE registros (
    id              INTEGER PRIMARY KEY,
    id_usuario      INTEGER,
    tipo_registro   TEXT,
    imagen          TEXT,
    enfermedad      TEXT,
    probabilidad    TEXT,
    descripcion     TEXT,
    tratamiento     TEXT,
    notas           TEXT,
    severidad       TEXT,
    calificacion    TEXT,
    mensaje         TEXT,
    leido           TEXT,
    enlace          TEXT
)
""")
# Deficiencias: probabilidad como TEXT (deberia ser REAL), sin FK a usuarios,
# mezcla diagnosticos y notificaciones en una sola tabla, sin timestamps,
# sin indices, calificacion como TEXT (deberia ser INTEGER), leido como TEXT

# ==============================================================================
# DATOS DE EJEMPLO - Version 1 (con inconsistencias tipicas)
# ==============================================================================

# Usuarios con datos de zona duplicados y passwords en texto plano
usuarios_data = [
    (1, 'Yersson', 'Calderon Romero', 'yersson_cr', 'yersson@gmail.com',
     'pass123',  # [D8] Contrasena en texto plano
     'Cusco', 'Cusco', 'San Sebastian', 'Peru', 'Av. La Cultura 123', 'admin'),
    (2, 'Maria', 'Quispe Huanca', 'mquispe', 'maria.quispe@gmail.com',
     'maria2024',
     'Cusco', 'Anta', 'Anta', 'Peru', None, 'user'),
    (3, 'Juan', 'Mamani Ccopa', 'jmamani', 'juan.mamani@hotmail.com',
     'juan1234',
     'Cusco', 'Urubamba', 'Urubamba', 'Peru', 'Jr. Arequipa 45', 'user'),
    (4, 'Rosa', 'Ttito Saire', 'rossita_t', 'rosa.ttito@gmail.com',
     'rosa999',
     'Cusco', 'Calca', 'Pisac', 'Peru', None, 'user'),
    (5, 'Carlos', 'Huanca Quispe', 'carlosH', 'carlos.huanca@gmail.com',
     'carlos2025',
     'Cusco', 'Cusco', 'Wanchaq', 'Peru', 'Calle Lima 78', 'user'),
    # [D6] Email duplicado - no hay restriccion UNIQUE
    (6, 'Pedro', 'Apaza Lima', 'pedro_a', 'maria.quispe@gmail.com',
     'pedro555',
     'Cusco', 'Espinar', 'Espinar', 'Peru', None, 'user'),
    # [D6] Usuario duplicado - no hay restriccion UNIQUE
    (7, 'Luis', 'Condori Puma', 'jmamani',
     'luis.condori@outlook.com', 'luis007',
     'Cusco', 'Canchis', 'Sicuani', 'Peru', None, 'user'),
]

cursor.executemany("""
INSERT INTO usuarios VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
""", usuarios_data)

# Registros: mezcla de diagnosticos y notificaciones en una sola tabla
# con tipos de datos incorrectos
registros_data = [
    # Diagnosticos (tipo_registro = 'diagnostico')
    (1,  1, 'diagnostico', 'uploads/hoja1.jpg', 'Tizón Tardío Papa',
     '87.5',  # [D4] probabilidad como TEXT
     'Enfermedad causada por Phytophthora infestans',
     'Aplicar fungicida con metalaxil',
     'Hoja del sector norte', 'grave', '4', None, None, None),
    (2,  1, 'diagnostico', 'uploads/hoja2.jpg', 'Mancha Foliar Maíz',
     '72.3',
     'Causada por Cercospora zeae-maydis',
     'Rotacion de cultivos, fungicidas preventivos',
     None, 'moderado', None, None, None, None),
    (3,  2, 'diagnostico', 'uploads/img_rosa.jpg', 'Oidio Tomate',
     '91',  # [D4] sin decimales, inconsistente
     'Hongo Leveillula taurica en hojas',
     'Azufre micronizado, eliminar hojas afectadas',
     'Cultivo de invernadero sector 2', 'leve', '5', None, None, None),
    (4,  3, 'diagnostico', 'uploads/hoja_juan.png', 'Sano',
     'muy alto',  # [D4] valor no numerico
     'Planta sin signos visibles de enfermedad',
     'Continuar monitoreo preventivo',
     None, None, None, None, None, None),
    (5,  4, 'diagnostico', 'uploads/pisac_001.jpg', 'Roya Amarilla Trigo',
     '65.8',
     'Puccinia striiformis f. sp. tritici',
     'Fungicidas triazol, variedad resistente',
     'Campo de la comunidad Pisac', 'moderado', '3', None, None, None),
    (6,  2, 'diagnostico', 'uploads/ant2.jpg', 'Tizón Tardío Papa',
     '88.1',
     'Phytophthora infestans con alta humedad',
     'Aplicar Mancozeb preventivo',
     None, 'grave', None, None, None, None),
    (7,  5, 'diagnostico', 'uploads/wanchaq01.jpg', 'Virus Mosaico Pepino',
     '55',
     'Cucumovirus en cucurbitaceas',
     'Control de afidos vectores, eliminar plantas',
     'Sector invernadero municipal', 'grave', '2', None, None, None),
    # Notificaciones mezcladas en la MISMA TABLA [D7]
    (8,  1, 'notificacion', None, None, None, None, None, None, None, None,
     'Alerta: Brote de Tizon Tardio en zona Anta',
     '0',  # [D7] leido como TEXT en vez de BOOLEAN
     '/alertas/tizon'),
    (9,  2, 'notificacion', None, None, None, None, None, None, None, None,
     'Nuevo diagnostico disponible en tu historial',
     '1',
     '/historial'),
    (10, 3, 'notificacion', None, None, None, None, None, None, None, None,
     'Actualizacion del sistema disponible',
     '0',
     '/noticias'),
    # Registro sin usuario valido [D1] - no hay FK que lo impida
    (11, 999, 'diagnostico', 'uploads/huerfano.jpg', 'Enfermedad Desconocida',
     '30.0',
     'No identificada',
     'Consulte a un especialista',
     None, None, None, None, None, None),
    # Registro con id_usuario NULL [D5]
    (12, None, 'diagnostico', 'uploads/anonimo.jpg', 'Alternaria Tomate',
     '78.9',
     'Alternaria solani',
     'Eliminar hojas infectadas, aplicar fungicida',
     None, 'moderado', None, None, None, None),
]

cursor.executemany("""
INSERT INTO registros VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
""", registros_data)

conn.commit()

# ==============================================================================
# REPORTE DE DEFICIENCIAS
# ==============================================================================
print("=" * 60)
print("  PlantAndes DB v1 - GENERADA CON EXITO")
print("=" * 60)
print(f"\nArchivo: {DB_PATH}")

# Verificar inconsistencias que el V2 resuelve
cursor.execute("SELECT correo, COUNT(*) FROM usuarios GROUP BY correo HAVING COUNT(*) > 1")
dup_emails = cursor.fetchall()

cursor.execute("SELECT usuario, COUNT(*) FROM usuarios GROUP BY usuario HAVING COUNT(*) > 1")
dup_users = cursor.fetchall()

cursor.execute("SELECT id FROM registros WHERE id_usuario NOT IN (SELECT id FROM usuarios) AND id_usuario IS NOT NULL")
orphan_records = cursor.fetchall()

cursor.execute("SELECT id FROM registros WHERE id_usuario IS NULL")
null_records = cursor.fetchall()

print("\n[DEFICIENCIAS DETECTADAS EN V1]")
print(f"  [D1] Sin FK: {len(orphan_records)} registro(s) huerfano(s) (id_usuario inexistente)")
print(f"  [D2] Sin normalizacion: datos de zona repetidos en cada usuario")
print(f"  [D3] Sin indices: 0 indices definidos")
print(f"  [D4] Tipo incorrecto: 'probabilidad' almacenada como TEXT")
print(f"  [D5] Sin NOT NULL: {len(null_records)} registro(s) con id_usuario NULL")
print(f"  [D6] Sin UNIQUE: {len(dup_emails)} email(s) duplicado(s), {len(dup_users)} usuario(s) duplicado(s)")
print(f"  [D7] Sin normalizacion: diagnosticos y notificaciones en 1 sola tabla")
print(f"  [D8] Contrasenas en texto plano (sin hash)")
print(f"  [D9] Sin timestamps en registros")
print(f"  [D10] Nombres de columnas inconsistentes")

print("\n[TABLAS CREADAS]")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
for t in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {t[0]}")
    count = cursor.fetchone()[0]
    print(f"  - {t[0]}: {count} filas")

print("\n[INDICES]")
cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
indexes = cursor.fetchall()
print(f"  Total indices: {len(indexes)} (ninguno definido por el desarrollador)")

conn.close()
print("\n[OK] Base de datos V1 lista para comparacion con V2.")
