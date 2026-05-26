"""
verify_gpu.py
Run this after setup to confirm everything is installed correctly.
Usage: python verify_gpu.py
"""

import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("\n=== Face Anti-Spoofing — Environment Check ===\n")

checks = [
    ("Python version",   lambda: sys.version.split()[0]),
    ("PyTorch",          lambda: __import__("torch").__version__),
    ("CUDA available",   lambda: str(__import__("torch").cuda.is_available())),
    ("GPU name",         lambda: __import__("torch").cuda.get_device_name(0)),
    ("CUDA version",     lambda: __import__("torch").version.cuda),
    ("timm",             lambda: __import__("timm").__version__),
    ("OpenCV",           lambda: __import__("cv2").__version__),
    ("albumentations",   lambda: __import__("albumentations").__version__),
    ("PyTorch Lightning",lambda: __import__("pytorch_lightning").__version__),
    ("WandB",            lambda: __import__("wandb").__version__),
    ("Gradio",           lambda: __import__("gradio").__version__),
    ("ONNX",             lambda: __import__("onnx").__version__),
]

passed = sum(check(name, fn) for name, fn in checks)
total  = len(checks)

print(f"\n{'='*46}")
print(f"  {passed}/{total} checks passed")

if passed == total:
    print("  All good! You're ready to train.")
else:
    print("  Fix the failed items, then re-run this script.")
print(f"{'='*46}\n")
