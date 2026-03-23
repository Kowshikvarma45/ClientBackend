import asyncio
import pandas as pd
import numpy as np
from processing import DataProcessor
from inference import ModelInference
import random
import os

async def compare_simulated_trends():
    print("--- START: compare_simulated_trends ---")
    dp = DataProcessor(api_key="test")
    inf = ModelInference()
    
    # Load a real model ID
    # Based on previous list_dir output:
    model_id = "af65b813-a7fc-49aa-90e8-722a9219afd3"
    success = inf.load_model(f"saved_models/{model_id}")
    if not success:
        print(f"❌ Could not load model {model_id}. Using default (will return dummy).")
    
    symbols = ["AAPL", "TSLA"]
    results = {}
    
    for symbol in symbols:
        print(f"Simulating for {symbol}...")
        n = 150 # More data to ensure MA20+ etc work
        if symbol == "AAPL":
            # Stable upward
            close = np.linspace(150, 160, n) + np.random.normal(0, 0.2, n)
            volume = np.linspace(1000, 1100, n)
        else:
            # Volatile upward
            close = np.linspace(200, 250, n) + np.random.normal(0, 10, n)
            volume = np.linspace(5000, 10000, n)
            
        df = pd.DataFrame({
            'close': close,
            'volume': volume,
            'date': pd.date_range(start='2023-01-01', periods=n)
        })
        
        # 1. Feature Prep
        scaled_data, stats = dp.prepare_numerical_features(df)
        
        # 2. Sequence
        TIMESTEPS = 60
        seq = scaled_data[-TIMESTEPS:]
        input_features = seq.reshape(1, TIMESTEPS, -1)
        
        # 3. Forecast
        res = inf.predict_forecast(input_features, days=7)
        norm_returns = res["forecast_returns"]
        
        if not norm_returns:
            print(f"❌ No returns for {symbol}")
            results[symbol] = [0]*7
            continue
            
        # 4. Price construction (with noise from history)
        last_close = float(df['close'].iloc[-1])
        hist_vol = float(df['close'].pct_change().std())
        if np.isnan(hist_vol): hist_vol = 0.02
        
        forecast_prices = []
        curr = last_close
        
        # Use a fixed seed for noise to make comparison focused on model+scaling
        random.seed(42) 
        
        for ret in norm_returns:
            # De-norm using index 0
            denorm_ret = (ret * inf.model_std[0]) + inf.model_mean[0]
            # Add noise (simulating main.py logic)
            noise = random.uniform(-hist_vol * 0.2, hist_vol * 0.2)
            denorm_ret += noise
            curr = curr * (1 + denorm_ret)
            forecast_prices.append(curr)
            
        pct_change = [(p - last_close)/last_close * 100 for p in forecast_prices]
        results[symbol] = pct_change
        print(f"{symbol} 7-day trend (%): {['%.2f' % x for x in pct_change]}")

    print("\nTrend Comparison:")
    if results["AAPL"] and results["TSLA"]:
        diff = np.abs(np.array(results["AAPL"]) - np.array(results["TSLA"]))
        print(f"Abs Diff between AAPL and TSLA trends: {['%.4f' % x for x in diff]}")
        
        if np.sum(diff) > 0.001:
             print(f"\n✅ Trends show diversity (Sum of diff: {np.sum(diff):.4f})")
        else:
             print("\n❌ Trends are still too similar.")
    
    print("--- END: compare_simulated_trends ---")

if __name__ == "__main__":
    asyncio.run(compare_simulated_trends())
