# scripts/hu25_pipeline.py
"""
HU-25 Pipeline:
- Guarda modelo original (si existe)
- Aplica pruning (poda) L1 unstructured
- (Opcional) fine-tune rápido (si proporcionas dataloaders)
- Aplica dynamic quantization
- Exporta a TorchScript y ONNX
- Quantiza ONNX (dynamic int8) con onnxruntime-tools
- Benchmarks (latencia y tamaños)
- Guarda resultados en outputs/hu25_results.csv y outputs/hu25_report.html
"""

import os, time, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import copy
import numpy as np
import pandas as pd

# ONNX quant tools
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ----------------- CONFIG -----------------
MODEL_PATH = MODELS_DIR / "model_original.pth"   # si tienes un .pth, ponlo aquí
INPUT_SHAPE = (1, 1, 28, 28)  # AJUSTA según tu modelo: (B,C,H,W) / o (1,784) para FC
USE_CUDA = False  # si quieres medir GPU, poner True (y tener CUDA)
PRUNE_AMOUNT = 0.30  # 30% por defecto
FINE_TUNE = False    # si pones True, el script intenta hacer fine-tune (sección TODO)
# ------------------------------------------

# ----- Ejemplo de modelo (reemplaza por tu clase en load_model())
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16*7*7, 64)
        self.fc2 = nn.Linear(64, 2)
        self.pool = nn.MaxPool2d(2,2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ----------------- helpers -----------------
def sizeof(path: Path):
    return path.stat().st_size / 1e6 if path.exists() else None

def benchmark_torch(model, example_input, n_warmup=10, n_iter=200, use_cuda=False):
    model.eval()
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        example_input = example_input.cuda()
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(example_input)
        if use_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(example_input)
        if use_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_iter

def benchmark_torchscript(ts_path, example_input, n_iter=200, use_cuda=False):
    m = torch.jit.load(str(ts_path)).eval()
    x = example_input
    if use_cuda and torch.cuda.is_available():
        m = m.cuda(); x = x.cuda(); torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(10): _ = m(x)
        if use_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter): _ = m(x)
        if use_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1-t0)/n_iter

def benchmark_onnx(onnx_path, example_input_np, n_iter=200):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    # warmup
    for _ in range(10): _ = sess.run(None, {input_name: example_input_np})
    t0 = time.perf_counter()
    for _ in range(n_iter): _ = sess.run(None, {input_name: example_input_np})
    t1 = time.perf_counter()
    return (t1-t0)/n_iter

# --------------- load model ---------------
def load_model_or_default():
    """
    Intenta cargar model_original.pth como state_dict en tu clase de modelo.
    Si no existe o no es compatible, devuelve el SimpleCNN de ejemplo.
    Si tienes tu propia clase: reemplaza SimpleCNN por tu clase y la carga.
    """
    model = SimpleCNN()
    if MODEL_PATH.exists():
        try:
            state = torch.load(MODEL_PATH, map_location='cpu')
            # si es state_dict
            if isinstance(state, dict):
                model.load_state_dict(state)
                print("Cargado modelo desde", MODEL_PATH)
            else:
                # si guardaste el objeto completo (no recomendado)
                model = state
                print("Cargado objeto de modelo completo desde", MODEL_PATH)
        except Exception as e:
            print("No se pudo cargar model_original.pth en SimpleCNN. Usando modelo por defecto. Error:", e)
    else:
        print("No se encontró model_original.pth. Usando modelo por defecto (weights aleatorias).")
    return model

# --------------- pipeline ---------------
def main():
    print("HU-25 pipeline - inicio")
    model = load_model_or_default()
    example_input = torch.randn(*INPUT_SHAPE)  # ajusta INPUT_SHAPE si es necesario
    example_input_np = example_input.cpu().numpy().astype(np.float32)

    # 1) guardar original (state_dict)
    orig_path = MODELS_DIR / "model_original_saved.pth"
    try:
        torch.save(model.state_dict(), orig_path)
        print("Guardado original:", orig_path, "size(MB):", sizeof(orig_path))
    except Exception as e:
        print("No se pudo guardar state_dict:", e)

    # baseline benchmark (FP32) - PyTorch
    t_baseline = benchmark_torch(model, example_input, n_warmup=10, n_iter=100, use_cuda=USE_CUDA)
    print("Baseline latency (ms):", t_baseline*1000)

    # 2) PRUNING
    print(f"Aplicando poda L1 unstructured (amount={PRUNE_AMOUNT}) ...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            try:
                prune.l1_unstructured(module, name='weight', amount=PRUNE_AMOUNT)
                prune.remove(module, 'weight')  # hace permanente la poda
                print("Pruned:", name)
            except Exception as e:
                print("Prune failed for", name, e)

    pruned_path = MODELS_DIR / "model_pruned.pth"
    torch.save(model.state_dict(), pruned_path)
    print("Guardado podado:", pruned_path, "size(MB):", sizeof(pruned_path))

    # TODO: si quieres fine-tune, aquí haz un entrenamiento corto para recuperar accuracy
    if FINE_TUNE:
        print("Fine-tune no implementado automáticamente. Agrega tu loop de entrenamiento aquí.")

    # 3) Dynamic quantization (PyTorch) - recomendado para capas Linear
    print("Aplicando dynamic quantization (PyTorch)...")
    try:
        model_cpu = copy.deepcopy(model).cpu()
        model_q = torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
        q_path = MODELS_DIR / "model_quantized.pth"
        torch.save(model_q.state_dict(), q_path)
        print("Guardado quantized (state_dict):", q_path, "size(MB):", sizeof(q_path))
    except Exception as e:
        print("Quantization failed:", e)
        model_q = model_cpu

    # 4) TorchScript export
    ts_path = MODELS_DIR / "model_optim_ts.pt"
    try:
        model_q.eval()
        traced = torch.jit.trace(model_q, example_input)
        traced.save(str(ts_path))
        print("TorchScript guardado:", ts_path, "size(MB):", sizeof(ts_path))
    except Exception as e:
        print("TorchScript export falló:", e)

    # 5) ONNX export (desde modelo pruned FP32)
    onnx_path = MODELS_DIR / "model_optim.onnx"
    try:
        model.eval()
        torch.onnx.export(model, example_input, str(onnx_path),
                          input_names=['input'], output_names=['output'],
                          opset_version=11, do_constant_folding=True)
        print("ONNX guardado:", onnx_path, "size(MB):", sizeof(onnx_path))
        # quantize ONNX dynamic
        onnx_q_path = MODELS_DIR / "model_optim_quant_onnx.onnx"
        quantize_dynamic(str(onnx_path), str(onnx_q_path), weight_type=QuantType.QInt8)
        print("ONNX quantizado guardado:", onnx_q_path, "size(MB):", sizeof(onnx_q_path))
    except Exception as e:
        print("ONNX export/quant failed:", e)
        onnx_path = None
        onnx_q_path = None

    # 6) Benchmarks comparativos
    results = []
    results.append({"variant":"baseline_pytorch_fp32", "size_mb": sizeof(orig_path), "latency_ms": t_baseline*1000})

    try:
        t_pruned = benchmark_torch(model, example_input, n_warmup=5, n_iter=100, use_cuda=USE_CUDA)
        results.append({"variant":"pruned_pytorch_fp32", "size_mb": sizeof(pruned_path), "latency_ms": t_pruned*1000})
    except Exception as e:
        print("Benchmark pruned failed:", e)

    try:
        t_quant = benchmark_torch(model_q, example_input, n_warmup=5, n_iter=100, use_cuda=USE_CUDA)
        results.append({"variant":"quantized_pytorch", "size_mb": sizeof(MODELS_DIR / 'model_quantized.pth'), "latency_ms": t_quant*1000})
    except Exception as e:
        print("Benchmark quantized failed:", e)

    try:
        if ts_path.exists():
            t_ts = benchmark_torchscript(ts_path, example_input, n_iter=100, use_cuda=USE_CUDA)
            results.append({"variant":"torchscript", "size_mb": sizeof(ts_path), "latency_ms": t_ts*1000})
    except Exception as e:
        print("Benchmark TorchScript failed:", e)

    try:
        if onnx_path is not None and (MODELS_DIR / "model_optim.onnx").exists():
            t_onnx = benchmark_onnx(MODELS_DIR / "model_optim.onnx", example_input.cpu().numpy().astype('float32'), n_iter=100)
            results.append({"variant":"onnx_fp32", "size_mb": sizeof(MODELS_DIR / "model_optim.onnx"), "latency_ms": t_onnx*1000})
        if onnx_q_path is not None and (MODELS_DIR / "model_optim_quant_onnx.onnx").exists():
            t_onnx_q = benchmark_onnx(MODELS_DIR / "model_optim_quant_onnx.onnx", example_input.cpu().numpy().astype('float32'), n_iter=100)
            results.append({"variant":"onnx_int8", "size_mb": sizeof(MODELS_DIR / "model_optim_quant_onnx.onnx"), "latency_ms": t_onnx_q*1000})
    except Exception as e:
        print("Benchmark ONNX failed:", e)

    # 7) Guardar resultados
    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "hu25_results.csv"
    df.to_csv(csv_path, index=False)
    print("Resultados guardados en:", csv_path)

    # 8) Generar HTML simple
    html = "<html><head><meta charset='utf-8'><title>HU-25 Report</title></head><body>"
    html += "<h1>HU-25 - Optimización del modelo</h1>"
    html += "<h2>Comparativa</h2>"
    html += df.to_html(index=False)
    html += "<h2>Model files (models/)</h2><ul>"
    for f in MODELS_DIR.iterdir():
        html += f"<li>{f.name} - {sizeof(f):.3f} MB</li>"
    html += "</ul></body></html>"
    html_path = OUT_DIR / "hu25_report.html"
    html_path.write_text(html, encoding='utf-8')
    print("HTML report guardado en:", html_path)

    print("HU-25 pipeline - fin")

if __name__ == "__main__":
    main()
    