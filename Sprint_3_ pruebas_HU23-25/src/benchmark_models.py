# scripts/benchmark_models.py
import os, time, torch, onnxruntime as ort, numpy as np

def bench_pytorch(model, x, n_iter=100):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(10):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(x)
        t1 = time.perf_counter()
    return (t1-t0)/n_iter

def bench_torchscript(ts_path, x_np, n_iter=100):
    m = torch.jit.load(ts_path).eval()
    x = torch.from_numpy(x_np)
    with torch.no_grad():
        for _ in range(10):
            _ = m(x)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = m(x)
        t1 = time.perf_counter()
    return (t1-t0)/n_iter

def bench_onnx(onnx_path, x_np, n_iter=100):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    # warmup
    for _ in range(10):
        _ = sess.run(None, {input_name: x_np})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = sess.run(None, {input_name: x_np})
    t1 = time.perf_counter()
    return (t1-t0)/n_iter

if __name__ == "__main__":
    # Ajusta la shape a tu modelo
    x_np = np.random.randn(1, 784).astype(np.float32)
    # Ejemplo: benchmark onnx
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_optim.onnx")
    if os.path.exists(onnx_path):
        t = bench_onnx(onnx_path, x_np, n_iter=100)
        print("ONNX avg ms:", t*1000)
    else:
        print("ONNX model not found:", onnx_path)
