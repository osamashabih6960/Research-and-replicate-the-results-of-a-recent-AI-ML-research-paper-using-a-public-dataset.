# src/utils.py
import os, json
def save_json(d, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
