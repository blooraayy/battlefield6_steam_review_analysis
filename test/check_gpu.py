import torch

print("=== Comprobación de GPU ===\n")

print(f"Versión de PyTorch : {torch.__version__}")
print(f"CUDA disponible    : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nNo se detecta GPU. El script usará CPU.")
else:
    print(f"Número de GPUs     : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"\n  GPU {i}: {props.name}")
        print(f"    VRAM total : {vram_gb:.1f} GB")
        print(f"    Compute    : {props.major}.{props.minor}")
