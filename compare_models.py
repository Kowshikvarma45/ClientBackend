import torch
import numpy as np
import os
from inference import ModelInference

def compare_models(model1_id, model2_id):
    inf = ModelInference()
    
    # Dummy input (zeros)
    dummy_input = np.zeros((1, 60, 2))
    
    print(f"--- Comparing {model1_id} and {model2_id} ---")
    
    inf.load_model(f"saved_models/{model1_id}")
    res1 = inf.predict_forecast(dummy_input, days=7)
    m1_mean = inf.model_mean
    m1_std = inf.model_std
    
    inf.load_model(f"saved_models/{model2_id}")
    res2 = inf.predict_forecast(dummy_input, days=7)
    m2_mean = inf.model_mean
    m2_std = inf.model_std
    
    print(f"Model 1 Mean: {m1_mean}")
    print(f"Model 2 Mean: {m2_mean}")
    print(f"Model 1 Forecast (Normalized): {res1['forecast_returns'][:3]}")
    print(f"Model 2 Forecast (Normalized): {res2['forecast_returns'][:3]}")
    
    # Compare raw weights (first layer weights)
    inf.load_model(f"saved_models/{model1_id}")
    w1 = inf.lstm.lstm.weight_ih_l0.detach().numpy().copy()
    
    inf.load_model(f"saved_models/{model2_id}")
    w2 = inf.lstm.lstm.weight_ih_l0.detach().numpy().copy()
    
    diff = np.sum(np.abs(w1 - w2))
    print(f"Weight diff (LSTM first layer): {diff}")
    
    if diff < 1e-6:
        print("❌ CRITICAL: Models have identical weights!")
    else:
        print("✅ Models have different weights.")

if __name__ == "__main__":
    # Use two IDs from the list
    id1 = "af65b813-a7fc-49aa-90e8-722a9219afd3"
    id2 = "de9ec5a5-5bb1-4782-ba4d-e640546fa6e3"
    compare_models(id1, id2)
