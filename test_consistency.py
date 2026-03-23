import numpy as np
import hashlib
from inference import ModelInference

def test_consistency():
    print("--- START: test_consistency ---")
    inf = ModelInference()
    symbol = "AAPL"
    date_str = "2026-02-23"
    seed_val = int(hashlib.md5((symbol + date_str).encode()).hexdigest(), 16) % (2**32)
    
    input_features = np.random.randn(1, 60, 5)
    
    # Call 1
    res1 = inf.predict_forecast(input_features, days=7, seed=seed_val)
    # Call 2
    res2 = inf.predict_forecast(input_features, days=7, seed=seed_val)
    
    print(f"Call 1 First Return: {res1['forecast_returns'][0]:.6f}")
    print(f"Call 2 First Return: {res2['forecast_returns'][0]:.6f}")
    
    assert res1['forecast_returns'] == res2['forecast_returns']
    assert res1['confidence'] == res2['confidence']
    print("✅ Seed-based consistency works! (Identical results across calls)")

def test_trend_aware_simulation():
    print("\n--- START: test_trend_aware_simulation ---")
    inf = ModelInference()
    
    # Simulate a STRONG UPWARD trend (+5% avg returns)
    upward_features = np.ones((1, 60, 5)) * 0.05
    res_up = inf.predict_forecast(upward_features, days=7, seed=123)
    avg_sim_up = np.mean(res_up['forecast_returns'])
    
    # Simulate a STRONG DOWNWARD trend (-5% avg returns)
    downward_features = np.ones((1, 60, 5)) * -0.05
    res_down = inf.predict_forecast(downward_features, days=7, seed=123)
    avg_sim_down = np.mean(res_down['forecast_returns'])
    
    print(f"Avg Simulated Return (Upward Trend): {avg_sim_up:.6f}")
    print(f"Avg Simulated Return (Downward Trend): {avg_sim_down:.6f}")
    
    assert avg_sim_up > avg_sim_down
    assert avg_sim_up > 0
    assert avg_sim_down < 0
    print("✅ Simulation is trend-aware! (Reflects historical bias)")

if __name__ == "__main__":
    test_consistency()
    test_trend_aware_simulation()
