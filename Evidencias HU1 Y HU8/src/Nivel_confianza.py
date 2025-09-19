import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


# 1. Cargar un modelo preentrenado (ejemplo: ResNet18)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# 2. Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3. Cargar una imagen de prueba
image_path = r"D:/Universidad/Cursos 9no semestre/Taller de proyectos 1/Evidencias HU1 Y HU8/images.jpg"

image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)  # batch size = 1

# 4. Pasar la imagen por el modelo
output = model(image_tensor)

# 5. Aplicar Softmax → obtener probabilidades
probabilidades = F.softmax(output, dim=1)

# 6. Obtener predicción principal y nivel de confianza
confianza, prediccion = torch.max(probabilidades, 1)
confianza = confianza.item() * 100  # convertir a porcentaje
prediccion = prediccion.item()

# 7. Validar confianza
if confianza < 50:
    print(f"⚠️ La predicción no es confiable ({confianza:.2f}%). Por favor suba otra imagen.")
else:
    print(f"✅ Predicción: {prediccion.item()}, Confianza: {confianza.item()*100:.2f}%")

