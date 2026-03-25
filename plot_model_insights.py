import os
import argparse
import asyncio
import asyncpg
import torch

# Force CPU to avoid OOM since the main server already holds VRAM
torch.cuda.is_available = lambda: False

import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import project modules
from processing import DataProcessor
from inference import ModelInference

async def get_latest_model_id_for_symbol(symbol: str) -> str:
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stocks")
    try:
        pool = await asyncpg.create_pool(db_url)
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                'SELECT id FROM "Model" WHERE symbol = $1 AND status = $2 ORDER BY "createdAt" DESC LIMIT 1',
                symbol, 'READY'
            )
        await pool.close()
        return row['id'] if row else None
    except Exception as e:
        print(f"DB Error: {e}")
        return None

def setup_data_and_model(symbol: str, days_foresight: int = 7):
    print(f"Loading data and model for {symbol}...")
    
    # Needs API Key for data processor, mock one since we depend on yfinance
    data_processor = DataProcessor(api_key="demo")
    
    # 1. Fetch historical data (using thread loop approach manually since we're in sync context)
    loop = asyncio.get_event_loop()
    df, news, currency = loop.run_until_complete(data_processor.fetch_data(symbol, "max"))
    
    if df is None or len(df) == 0:
        raise ValueError(f"No data available for symbol {symbol}")
        
    print(f"Data fetched: {len(df)} records for {symbol}.")
    
    # 2. Get scaled sequences
    scaled_data, stats = data_processor.prepare_numerical_features(df)
    
    # 3. Find the freshest model for the symbol
    loop = asyncio.get_event_loop()
    model_id = loop.run_until_complete(get_latest_model_id_for_symbol(symbol))
    
    if not model_id:
        print(f"Database did not have a READY model for {symbol}. Looking in saved_models loosely...")
        model_dir = "saved_models"
        available_models = [m.split("_lstm.pth")[0] for m in os.listdir(model_dir) if m.endswith("_lstm.pth")]
        if not available_models:
            raise FileNotFoundError(f"No trained models found in {model_dir}")
        model_id = available_models[-1]
        
    print(f"Using model ID {model_id} for symbol {symbol}")
    model_dir = "saved_models"
    model_path = os.path.join(model_dir, str(model_id))
    
    inference = ModelInference()
    success = inference.load_model(model_path)
    
    if not success:
        raise RuntimeError("Failed to load model weights.")
        
    return df, scaled_data, stats, inference

def generate_actual_vs_predicted(symbol: str, df: pd.DataFrame, scaled_data: np.ndarray, stats: dict, inference: ModelInference, past_days: int = 60, output_dir: str = "."):
    """
    Validation Plot: Actual vs Predicted Price Scatter or Line Plot
    """
    print("Generating Actual vs Predicted Price graph...")
    
    # We will simulate the last N days where we test the model's 1-day foresight
    # That means for each day t in the last N days, we use the 60 window ending at t-1 to predict t
    
    actual_prices = []
    predicted_prices = []
    dates = []
    
    seq_length = 60
    
    if len(scaled_data) < seq_length + past_days:
        past_days = len(scaled_data) - seq_length
        if past_days <= 0:
            raise ValueError("Not enough data to run sequence validation plot.")
            
    test_start_idx = len(scaled_data) - past_days
    
    mean_model = inference.model_mean
    std_model = inference.model_std
    
    for i in range(test_start_idx, len(scaled_data)):
        # End of the window
        seq = scaled_data[i-seq_length:i].reshape(1, seq_length, -1)
        
        # Actual price at time i
        actual_val = df['close'].iloc[i]
        
        res = inference.predict_forecast(seq, days=1, seed=42)
        norm_ret = res["forecast_returns"][0]
        
        # De-normalize 
        pred_ret = (norm_ret * std_model[0]) + mean_model[0]
        
        # Calculate predicted price based on ACTUAL previous day's close
        prev_actual = df['close'].iloc[i-1]
        pred_val = prev_actual * (1 + pred_ret)
        
        actual_prices.append(actual_val)
        predicted_prices.append(pred_val)
        dates.append(df['date'].iloc[i])
        
    # Plotting
    plt.figure(figsize=(12, 6), facecolor="#1E1E1E")
    ax = plt.gca()
    ax.set_facecolor("#1E1E1E")
    
    plt.plot(dates, actual_prices, color="#00FFCC", label="Actual Price", linewidth=2)
    plt.plot(dates, predicted_prices, color="#FF0055", label="Predicted Price", linestyle="--", linewidth=2)
    
    plt.title(f"{symbol} - Actual vs Predicted Price (Last {past_days} Days)", color="white", fontsize=14)
    plt.xlabel("Date", color="white")
    plt.ylabel(f"Price", color="white")
    
    plt.xticks(color="white")
    plt.yticks(color="white")
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.legend(facecolor="#2A2A2A", edgecolor="white", labelcolor="white")
    plt.grid(color="#333333", linestyle=":", linewidth=1)
    
    # Actual vs Predicted Scatter Plot as sub-plot if we wanted, but Line is listed as #1 choice.
    # We will save it.
    output_path = os.path.join(output_dir, f"{symbol}_actual_vs_predicted.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved to {output_path}")

def generate_future_forecast(symbol: str, df: pd.DataFrame, scaled_data: np.ndarray, stats: dict, inference: ModelInference, output_dir: str = "."):
    """
    Generating Future Forecast Plot (Rolling Prediction)
    """
    print("Generating Future Forecast Plot...")
    
    seq_length = 60
    days_to_predict = 14 # 2 Weeks 
    
    # We use the absolute final 60 days
    last_seq = scaled_data[-seq_length:].reshape(1, seq_length, -1)
    
    # Run prediction
    res = inference.predict_forecast(last_seq, days=days_to_predict, seed=42)
    norm_prices = res["forecast_returns"]
    
    mean_model = inference.model_mean
    std_model = inference.model_std
    
    forecast_prices = []
    hist_vol = float(df['returns'].std())
    rng = random.Random(42)
    
    current_price = df['close'].iloc[-1]
    
    for norm_ret in norm_prices:
        pred_ret = (norm_ret * std_model[0]) + mean_model[0]
        noise_pct = rng.uniform(-hist_vol * 0.1, hist_vol * 0.1)
        pred_ret = pred_ret + noise_pct
        pred_ret = max(-0.15, min(0.15, pred_ret))
        
        current_price = current_price * (1 + pred_ret)
        forecast_prices.append(current_price)
        
    # Grab the last 30 days of historical to show the join
    hist_dates = list(df['date'].iloc[-30:])
    hist_prices = list(df['close'].iloc[-30:])
    
    # Generate future dates
    last_date = hist_dates[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_predict+1)]
    
    plt.figure(figsize=(12, 6), facecolor="#1E1E1E")
    ax = plt.gca()
    ax.set_facecolor("#1E1E1E")
    
    # Historical
    plt.plot(hist_dates, hist_prices, color="#AAAAAA", label="Historical Path", linewidth=2)
    
    # Forecast 
    # Link the lines properly
    join_dates = [hist_dates[-1]] + future_dates
    join_prices = [hist_prices[-1]] + forecast_prices
    plt.plot(join_dates, join_prices, color="#A455FF", label="Future Forecast", linestyle="-", linewidth=3)
    
    plt.title(f"{symbol} - Future Forecast Plot (Rolling Prediction)", color="white", fontsize=14)
    plt.xlabel("Date", color="white")
    plt.ylabel(f"Price", color="white")
    
    plt.xticks(color="white", rotation=45)
    plt.yticks(color="white")
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.legend(facecolor="#2A2A2A", edgecolor="white", labelcolor="white")
    plt.grid(color="#333333", linestyle=":", linewidth=1)
    
    output_path = os.path.join(output_dir, f"{symbol}_forecast.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model diagnostic graphs.")
    parser.add_argument("symbol", type=str, help="Stock Symbol (e.g., AAPL)")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    try:
        df, scaled_data, stats, inference = setup_data_and_model(symbol)
        generate_actual_vs_predicted(symbol, df, scaled_data, stats, inference, output_dir=args.out)
        generate_future_forecast(symbol, df, scaled_data, stats, inference, output_dir=args.out)
    except Exception as e:
        print(f"Error: {e}")
