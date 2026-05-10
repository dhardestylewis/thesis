from catboost import CatBoostClassifier
import pandas as pd

try:
    model = CatBoostClassifier(iterations=10, depth=16, verbose=0)
    print("[*] Depth 16 initialized successfully.")
except Exception as e:
    print(f"[!] Depth 16 failed: {e}")

try:
    model = CatBoostClassifier(iterations=10, depth=18, verbose=0)
    print("[*] Depth 18 initialized successfully.")
except Exception as e:
    print(f"[!] Depth 18 failed: {e}")

try:
    model = CatBoostClassifier(iterations=10, depth=20, verbose=0)
    print("[*] Depth 20 initialized successfully.")
except Exception as e:
    print(f"[!] Depth 20 failed: {e}")
