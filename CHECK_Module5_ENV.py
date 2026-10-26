# -*- coding: utf-8 -*-
import importlib
import sys

PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("tifffile", "tifffile"),
    ("imagecodecs", "imagecodecs"),
    ("cv2", "opencv-python"),
    ("PySide6", "PySide6"),
    ("matplotlib", "matplotlib"),
    ("networkx", "networkx"),
    ("imageio", "imageio"),
    ("imageio_ffmpeg", "imageio-ffmpeg"),
    ("skimage", "scikit-image"),
    ("tqdm", "tqdm"),
]

print("Python:", sys.version)
print("Executable:", sys.executable)
print()

failed = []
for modname, pipname in PACKAGES:
    try:
        mod = importlib.import_module(modname)
        ver = getattr(mod, "__version__", "OK")
        print(f"{modname:14s} {ver}")
    except Exception as exc:
        print(f"{modname:14s} FAILED: {exc}")
        failed.append((pipname, exc))

try:
    import numpy as np
    import tifffile
    nmaj = int(np.__version__.split('.')[0])
    parts = tifffile.__version__.split('.')
    year = int(parts[0]) if parts and parts[0].isdigit() else 9999
    if nmaj >= 2 and year < 2024:
        print("\n[ERROR] NumPy >= 2 with old tifffile is unsafe for this project.")
        failed.append(("numpy/tifffile compatibility", "incompatible versions"))
except Exception:
    pass

print()
if failed:
    print("ENV CHECK FAILED")
    raise SystemExit(1)

print("ENV CHECK PASSED")
