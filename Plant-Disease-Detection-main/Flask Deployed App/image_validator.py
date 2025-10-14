import cv2
import numpy as np

def check_blur_from_stream(image_stream, threshold=100.0):
    """
    Calcula la nitidez de una imagen usando la varianza del Laplaciano.
    Una puntuación baja indica una imagen borrosa.

    :param image_stream: El contenido de la imagen en bytes.
    :param threshold: Umbral por debajo del cual la imagen se considera borrosa.
    :return: Una tupla (puntuación de calidad, es_borrosa)
    """
    # Decodificar el flujo de bytes en una imagen de OpenCV
    image_array = np.frombuffer(image_stream, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Si la imagen no se pudo decodificar, devolvemos un valor seguro.
    if image is None:
        return 0.0, True

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Determinar si la imagen es borrosa
    is_blurry = laplacian_var < threshold

    return laplacian_var, is_blurry