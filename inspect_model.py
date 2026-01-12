import joblib
import sys

# Point to the file in the other directory
model_path = r"C:\Users\Shadow\.gemini\antigravity\brain\fe3cc05d-b1ed-4792-ae8d-494cf0e02a48\ai_candle_model.pkl"

try:
    print(f"Loading {model_path}...")
    data = joblib.load(model_path)
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        if 'features' in data:
            print(f"Features List: {data['features']}")
        if 'model' in data:
            print(f"Model type: {type(data['model'])}")
    else:
        print("Not a dict.")
except Exception as e:
    print(f"Error: {e}")
