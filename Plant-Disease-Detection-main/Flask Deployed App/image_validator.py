# ==============================================================================
# Módulo de Validación de Calidad de Imagen para PlantAndes
#
# Copyright (c) 2025 JherzonDev. Todos los derechos reservados.
#
# Autor: JherzonDev
#
# ==============================================================================

import cv2
import numpy as np

def check_blur_from_stream(image_stream, threshold=100.0):
    # 1. Decodificar el flujo de bytes en una imagen que OpenCV pueda procesar.
    image_array = np.frombuffer(image_stream, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Manejar el caso de que el archivo no sea una imagen válida.
    if image is None:
        return 0.0, True

    # 2. Convertir a escala de grises, ya que el Laplaciano opera sobre un solo canal.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 3. Calcular la varianza del Laplaciano.
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return laplacian_var, (laplacian_var < threshold)

# ==============================================================================
# Fin del módulo de validación.
# ==============================================================================