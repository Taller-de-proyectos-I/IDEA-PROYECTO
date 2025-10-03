import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 🔹 Modelo simple para ejemplo
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(20, 16)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Dataset ficticio (200 muestras, 20 features, 2 clases)
X = torch.randn(200, 20)
y = torch.randint(0, 2, (200,))

model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train_loss, val_loss, train_acc, val_acc = [], [], [], []

# 🔹 Entrenamiento simulado
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    # Guardamos métricas ficticias
    preds = torch.argmax(outputs, 1)
    acc = (preds == y).float().mean().item()
    train_loss.append(loss.item())
    val_loss.append(loss.item() * 1.1)   # simula mayor pérdida en validación
    train_acc.append(acc)
    val_acc.append(max(0, acc - 0.05))   # simula menor precisión en validación



# 🔹 Gráfico de pérdida y precisión
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="mayor pérdida en validación")
plt.xlabel("Épocas"); plt.ylabel("perdida"); plt.title("Pérdida por época")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="menor precisión en validación")
plt.xlabel("Épocas"); plt.ylabel("Precisión"); plt.title("Precisión por época")
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix

y_pred = preds.detach().numpy()
y_true = y.numpy()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sanos","Enfermos"], yticklabels=["Sanos","Enfermos"])
plt.xlabel("Prediccion"); plt.ylabel("Real"); plt.title("Matriz de Confusión")
plt.show()

from sklearn.metrics import roc_curve, auc

y_score = F.softmax(outputs, dim=1)[:,1].detach().numpy()  # probas de clase 1
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.show()



###############################################

# Guardar gráficas automáticamente

# 📊 Gráfico de pérdida y precisión
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="mayor pérdida en validación")
plt.xlabel("Épocas"); plt.ylabel("perdida"); plt.title("Pérdida por época")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="menor precisión en validación")
plt.xlabel("Épocas"); plt.ylabel("Precisión"); plt.title("Precisión por época")
plt.legend()
plt.tight_layout()
plt.savefig("Sprint_3_ pruebas_HU23-25/HU-23/metricas_epocas.png")
plt.savefig("Sprint_3_ pruebas_HU23-25/HU-23/metricas_epocas.pdf")
plt.show()

# 📊 Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Clase 0","Clase 1"], yticklabels=["Clase 0","Clase 1"])
plt.xlabel("Prediccion"); plt.ylabel("Real"); plt.title("Matriz de Confusión")
plt.tight_layout()
plt.savefig("Sprint_3_ pruebas_HU23-25/HU-23/matriz_confusion.png")
plt.savefig("Sprint_3_ pruebas_HU23-25/HU-23/matriz_confusion.pdf")
plt.show()

# 📊 Curva ROC
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("Sprint_3_ pruebas_HU23-25/HU-23/curva_ROC.png")
plt.savefig("Sprint_3_ pruebas_HU23-25/HU-23/curva_ROC.pdf")
plt.show()
