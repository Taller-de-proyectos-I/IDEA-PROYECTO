# ==============================================================================
# Script de Respaldo para PlantAndes
#
# Copyright (c) 2025 Yersson Calderon Romero. Todos los derechos reservados.
#
# Autor: Yersson Calderon Romero
#
# ==============================================================================

import os
import shutil
import datetime
import zipfile

def create_backup(source_paths, backup_dir, retention_days=7):
    # 1. Crear el directorio de respaldos si no existe
    os.makedirs(backup_dir, exist_ok=True)

    # 2. Definir el nombre del archivo de respaldo con fecha y hora
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    zip_filename = os.path.join(backup_dir, f"backup_{timestamp}.zip")

    print(f"Creando respaldo en: {zip_filename}")

    try:
        # 3. Crear el archivo ZIP y añadir los archivos/directorios
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for path in source_paths:
                if not os.path.exists(path):
                    print(f"ADVERTENCIA: La ruta '{path}' no existe y será omitida.")
                    continue

                if os.path.isfile(path):
                    # Si es un archivo, añadirlo directamente
                    zipf.write(path, os.path.basename(path))
                    print(f"  - Añadiendo archivo: {path}")
                elif os.path.isdir(path):
                    # Si es un directorio, añadir todo su contenido
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Crear una ruta relativa para no guardar la estructura completa del disco
                            relative_path = os.path.relpath(file_path, os.path.dirname(path))
                            zipf.write(file_path, relative_path)
                    print(f"  - Añadiendo directorio: {path}")

        print("¡Respaldo creado exitosamente!")

        # 4. Limpiar respaldos antiguos
        cleanup_old_backups(backup_dir, retention_days)

    except Exception as e:
        print(f"ERROR: Ocurrió un error al crear el respaldo: {e}")

def cleanup_old_backups(backup_dir, retention_days):
    """Elimina los respaldos que son más antiguos que el período de retención."""
    print(f"Limpiando respaldos con más de {retention_days} días de antigüedad...")
    now = datetime.datetime.now()
    cutoff_date = now - datetime.timedelta(days=retention_days)

    for filename in os.listdir(backup_dir):
        file_path = os.path.join(backup_dir, filename)
        if os.path.isfile(file_path) and filename.startswith('backup_') and filename.endswith('.zip'):
            try:
                file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mod_time < cutoff_date:
                    os.remove(file_path)
                    print(f"  - Eliminado respaldo antiguo: {filename}")
            except Exception as e:
                print(f"ERROR: No se pudo eliminar el archivo {filename}. Razón: {e}")

if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    # Rutas de los archivos y directorios importantes a respaldar
    PATHS_TO_BACKUP = [
        'plantandes.db',
        'plant_disease_model_2.pt',
        'static/uploads'
    ]
    # Directorio donde se guardarán los archivos .zip de respaldo
    BACKUP_DESTINATION = 'D:\\Backups_plant_dissease' 
    create_backup(PATHS_TO_BACKUP, BACKUP_DESTINATION)

# ==============================================================================
# Fin del script de respaldo.
# ==============================================================================