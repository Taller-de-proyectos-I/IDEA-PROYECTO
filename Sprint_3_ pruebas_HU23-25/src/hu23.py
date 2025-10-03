# scripts/hu23.py
"""
HU-23: Visualización de métricas
- Genera: pérdida/precisión por época, matriz de confusión, curva ROC (y AUC)
- Guarda PNG/PDF en outputs/ y genera un HTML simple con las imágenes.
- Si encuentra models/model_original.pth intentará cargarlo (espera una
    red PyTorch con método forward que acepte un tensor), sino simula datos.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# -------------------------
MODEL_PATH = MODELS_DIR / "model_original.pth"
# -------------------------

def simulate_training_and_preds():
    # Simula métricas y predicciones (binary) para probar el flujo
    epochs = 10
    train_loss = np.linspace(1.0, 0.3, epochs) + np.random.randn(epochs)*0.03
    val_loss   = train_loss + 0.05 + np.random.randn(epochs)*0.02
    train_acc  = np.linspace(0.55, 0.92, epochs)
    val_acc    = train_acc - 0.04 + np.random.randn(epochs)*0.02

    # Simula dataset de validación (binary)
    n = 200
    y_true = np.random.randint(0, 2, size=n)
    # probabilidades simuladas (mejor en positivos si y_true==1)
    y_score = np.clip(0.3 + 0.6*y_true + np.random.randn(n)*0.15, 0, 1)
    y_pred = (y_score >= 0.5).astype(int)

    return train_loss, val_loss, train_acc, val_acc, y_true, y_pred, y_score

# Ejemplo de carga sencilla de modelo (si lo provees, debe adaptarse a tu arquitectura)
def load_model_and_predict():
    # Este ejemplo espera un modelo que reciba un batch y devuelva logits (N,C)
    # Si tu modelo requiere transforms o shape distinta, edítalo aquí.
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(20, 2)
        def forward(self, x):
            return self.fc(x)

    # Si tienes un .pth con state_dict para tu clase, reemplaza DummyNet() por tu clase.
    model = DummyNet()
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        # intenta cargar state_dict, si falla asumimos que no es compatible
        if isinstance(state, dict):
            model.load_state_dict(state)
        else:
            print("El contenido de model_original.pth no parece ser un state_dict compatible con DummyNet.")
    except Exception as e:
        print("No se cargó modelo real (se usará simulación). Error:", e)
        return None

    model.eval()
    # Simula un val set de 200 muestras con 20 features (ajusta según tu modelo)
    X_val = torch.randn(200, 20)
    with torch.no_grad():
        logits = model(X_val)  # shape [N, C]
        probs = F.softmax(logits, dim=1).cpu().numpy()
        y_score = probs[:, 1] if probs.shape[1] >= 2 else probs[:, 0]
        y_pred = np.argmax(probs, axis=1)
        # simula etiquetas reales (sólo para flujo; ideal: cargar y_true reales)
        y_true = np.random.randint(0, probs.shape[1], size=X_val.shape[0])
    # For training curves, we don't have training history—so retornamos None y el resto
    return None, None, None, None, y_true, y_pred, y_score

# -------------------------
def plot_and_save_curves(train_loss, val_loss, train_acc, val_acc):
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
    p1 = OUTPUTS / "metricas_epocas.png"
    p2 = OUTPUTS / "metricas_epocas.pdf"
    plt.savefig(p1)
    plt.savefig(p2)
    plt.close()
    plt.show()
    return p1, p2

def plot_and_save_confusion(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    if labels is None:
        labels = [f"Clase {i}" for i in range(cm.shape[0])]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prediccion"); plt.ylabel("Real"); plt.title("Matriz de Confusión")
    plt.tight_layout()
    p1 = OUTPUTS / "matriz_confusion.png"
    p2 = OUTPUTS / "matriz_confusion.pdf"
    plt.savefig(p1)
    plt.savefig(p2)
    plt.close()
    plt.show()
    return p1, p2

def plot_and_save_roc(y_true, y_score):
    # Manejo binario; si multiclass, se pueden calcular curvas por clase (no implementado aquí)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("falsos positivos"); plt.ylabel("verdaderos positivos"); plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    p1 = OUTPUTS / "curva_ROC.png"
    p2 = OUTPUTS / "curva_ROC.pdf"
    plt.savefig(p1)
    plt.savefig(p2)
    plt.close()
    plt.show()
    return p1, p2, roc_auc

def generate_html_report(images_paths, metrics_summary):
    html = "<html><head><meta charset='utf-8'><title>Reporte HU-23</title></head><body>"
    html += "<h1>Reporte HU-23 - Visualización de métricas</h1>"
    html += "<h2>Resumen de métricas</h2><ul>"
    for k,v in metrics_summary.items():
        html += f"<li><strong>{k}:</strong> {v}</li>"
    html += "</ul>"
    html += "<h2>Gráficas</h2>"
    for p in images_paths:
        html += f"<div style='margin:20px 0'><img src='{p.name}' style='max-width:800px'><p>{p.name}</p></div>"
    html += "</body></html>"
    out_html = OUTPUTS / "report_HU23.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print("HTML generado en:", out_html)
    return out_html

def main():
    # 1) intenta cargar modelo real y predecir
    loaded = load_model_and_predict()
    if loaded is None:
        print("Usando simulación para métricas y predicciones.")
        train_loss, val_loss, train_acc, val_acc, y_true, y_pred, y_score = simulate_training_and_preds()
    else:
        # si load_model_and_predict devolvió real, su forma fue definida allí:
        # (en el ejemplo, load_model_and_predict devuelve None si no cargó)
        train_loss, val_loss, train_acc, val_acc, y_true, y_pred, y_score = loaded

    # Si no hay historial de entrenamiento, simulamos curvas sencillas
    if train_loss is None:
        train_loss = np.linspace(1.0, 0.4, 10)
        val_loss   = train_loss + 0.05
        train_acc  = np.linspace(0.55, 0.90, 10)
        val_acc    = train_acc - 0.03

    # 2) Plots y guardado
    p_metrics_png, p_metrics_pdf = plot_and_save_curves(train_loss, val_loss, train_acc, val_acc)
    p_cm_png, p_cm_pdf = plot_and_save_confusion(y_true, y_pred)
    p_roc_png, p_roc_pdf, roc_auc = plot_and_save_roc(y_true, y_score)

    # 3) metrics summary
    acc = accuracy_score(y_true, y_pred)
    metrics_summary = {
        "Exactitud (val)": f"{acc:.3f}",
        "AUC ROC": f"{roc_auc:.3f}",
        "N muestras (val)": len(y_true)
    }

    # 4) Gen HTML report (usa imágenes relativas en outputs)
    images = [p_metrics_png, p_cm_png, p_roc_png]
    html_path = generate_html_report(images, metrics_summary)

    print("Gráficas guardadas en:", p_metrics_png, p_cm_png, p_roc_png)
    print("Resumen:", metrics_summary)
    print("Fin HU-23")

if __name__ == "__main__":
    main()
