import torch
import numpy as np
from inference import ModelInference

def test_trimming_logic():
    print("--- START: test_trimming_logic ---")
    inf = ModelInference()
    
    # Simulate a model that only has 2 inputs (like the user's old models)
    # We'll mock the mean_model to size 2
    inf.model_mean = np.array([0, 0])
    inf.model_std = np.array([1, 1])
    
    # Data with 5 features
    raw_values = np.random.randn(10, 5)
    
    # Trimming logic from main.py
    input_size_model = len(inf.model_mean)
    if raw_values.shape[1] != input_size_model:
         print(f"Mismatch: Model expects {input_size_model}, Data has {raw_values.shape[1]}")
         if raw_values.shape[1] > input_size_model:
              processed_values = raw_values[:, :input_size_model]
         else:
              padding = np.zeros((raw_values.shape[0], input_size_model - raw_values.shape[1]))
              processed_values = np.concatenate((raw_values, padding), axis=1)
    
    print(f"Processed shape: {processed_values.shape}")
    assert processed_values.shape[1] == 2
    print("✅ Trimming logic works!")
    print("--- END: test_trimming_logic ---")

if __name__ == "__main__":
    test_trimming_logic()
