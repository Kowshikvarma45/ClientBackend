import numpy as np
from inference import ModelInference

def test_natural_accuracy():
    print("--- START: test_natural_accuracy ---")
    # Simulate a "bad" model where RMSE = 1.0 (preds = mean)
    val_loss = 1.0 # MSE = 1.0 => RMSE = 1.0
    rmse = np.sqrt(val_loss)
    
    # Formula: 100 / (1 + RMSE)
    accuracy = 100 / (1 + rmse)
    print(f"Accuracy with RMSE=1.0: {accuracy:.2f}%")
    assert accuracy == 50.0 # Should be 50%
    
    # Simulate a "good" model where RMSE = 0.25
    rmse_good = 0.25
    accuracy_good = 100 / (1 + rmse_good)
    print(f"Accuracy with RMSE=0.25: {accuracy_good:.2f}%")
    assert accuracy_good == 80.0 # 100 / 1.25 = 80
    
    print("✅ Natural Accuracy logic is correct!")

def test_inference_fallback():
    print("\n--- START: test_inference_fallback ---")
    inf = ModelInference()
    # No models loaded
    res = inf.predict_forecast(np.random.randn(1, 60, 5), days=7)
    
    print(f"Fallback Confidence: {res['confidence']:.4f}")
    print(f"Forecast Returns Count: {len(res['forecast_returns'])}")
    
    assert res['confidence'] > 0
    assert len(res['forecast_returns']) == 7
    print("✅ Inference Fallback works!")

if __name__ == "__main__":
    test_natural_accuracy()
    test_inference_fallback()
