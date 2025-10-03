# scripts/prune_quantize_export.py
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import time
import copy
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# ----- CONFIG -----
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Ajusta esto a tu clase de modelo si ya tienes una
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------
def save_state(model, path):
    torch.save(model.state_dict(), path)
    print("Saved:", path, "->", os.path.getsize(path)/1e6, "MB")

def load_example_model():
    model = SimpleNet()
    # adapta aquí si cargas tu modelo: model.load_state_dict(torch.load("models/model_original.pth"))
    return model

def benchmark_torch(model, input_tensor, n_warmup=10, n_iter=200, use_cuda=False):
    model.eval()
    if use_cuda:
        model.cuda(); input_tensor = input_tensor.cuda(); torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
    if use_cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = model(input_tensor)
    if use_cuda: torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter

def main():
    # 0) load/create model
    model = load_example_model()
    example_input = torch.randn(1, 784)

    # 1) save original state (baseline)
    orig_path = os.path.join(MODELS_DIR, "model_original.pth")
    save_state(model, orig_path)

    # baseline benchmark (FP32)
    t_baseline = benchmark_torch(model, example_input, n_warmup=10, n_iter=100)
    print(f"Baseline latency (ms): {t_baseline*1000:.3f}")

    # 2) PRUNING - prune 30% L1 unstructured on Linear/Conv layers
    print("Applying pruning (30%) ...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name="weight", amount=0.3)
            prune.remove(module, "weight")  # make pruning permanent
            print("Pruned:", name)

    pruned_path = os.path.join(MODELS_DIR, "model_pruned.pth")
    save_state(model, pruned_path)

    # Optional: you should fine-tune here on your dataset to recover accuracy.
    # (left as TODO: run short fine-tune if you have train loader)

    # 3) DYNAMIC QUANTIZATION (works well for Linear layers on CPU)
    print("Applying dynamic quantization...")
    model_cpu = model.cpu()
    model_q = torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
    q_path = os.path.join(MODELS_DIR, "model_quantized.pth")
    save_state(model_q, q_path)

    # 4) TorchScript export (trace) from quantized CPU model (if traceable)
    try:
        print("Exporting TorchScript (trace)...")
        model_q.eval()
        traced = torch.jit.trace(model_q, example_input)
        ts_path = os.path.join(MODELS_DIR, "model_optim_ts.pt")
        traced.save(ts_path)
        print("Saved TorchScript:", ts_path)
    except Exception as e:
        print("TorchScript tracing failed:", e)

    # 5) ONNX export (export FP32 pruned model; ONNX quantization can be applied afterwards)
    try:
        print("Exporting ONNX (FP32) ...")
        onnx_path = os.path.join(MODELS_DIR, "model_optim.onnx")
        # ensure model is in cpu and eval mode
        model.cpu().eval()
        torch.onnx.export(model, example_input, onnx_path,
                        input_names=["input"], output_names=["output"],
                        opset_version=11, do_constant_folding=True)
        print("Saved ONNX:", onnx_path, "size:", os.path.getsize(onnx_path)/1e6, "MB")

        # 6) Optional: quantize ONNX with onnxruntime (dynamic int8)
        onnx_q_path = os.path.join(MODELS_DIR, "model_optim_quant_onnx.onnx")
        print("Quantizing ONNX (dynamic int8)...")
        quantize_dynamic(onnx_path, onnx_q_path, weight_type=QuantType.QInt8)
        print("Saved ONNX quantized:", onnx_q_path, "size:", os.path.getsize(onnx_q_path)/1e6, "MB")

    except Exception as e:
        print("ONNX export/quantization failed:", e)

    # 7) Benchmark quantized PyTorch (if available)
    try:
        t_quant = benchmark_torch(model_q, example_input, n_warmup=10, n_iter=100)
        print(f"Quantized PyTorch latency (ms): {t_quant*1000:.3f}")
    except Exception as e:
        print("Benchmark quantized model failed:", e)

if __name__ == "__main__":
    main()
