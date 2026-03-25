from fastapi import FastAPI, HTTPException, Body, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import logging
import asyncpg
import asyncio
import numpy as np
from dotenv import load_dotenv
from processing import DataProcessor
from inference import ModelInference
from decision_engine import DecisionEngine
from training import train_model_task, training_progress

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables from the specific location
load_dotenv(os.path.join(os.path.dirname(__file__), "../stocks/db/.env"))

app = FastAPI(title="Stock Prediction API", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Config (Should normally be in .env)
# Using the default usually provided by Docker/local setup: postgresql://postgres:postgres@localhost:5432/stocks
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stocks")

db_pool = None

@app.on_event("startup")
async def startup():
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        logger.info("Connected to Database")
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

# Initialize modules
# Get API Key from env
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "MOCK_KEY")
print(f"Using Alpha Vantage Key: {str(ALPHA_VANTAGE_KEY)[:5]}...")

data_processor = DataProcessor(api_key=ALPHA_VANTAGE_KEY)
model_inference = ModelInference()
decision_engine = DecisionEngine()

class StockRequest(BaseModel):
    symbol: str
    period: str = "1mo" # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

class TrainRequest(BaseModel):
    symbol: str
    epochs: int = 50
    learning_rate: float = 0.001

@app.get("/")
def read_root():
    return {"message": "Stock Prediction API is running"}

@app.post("/predict")
async def predict_stock(request: StockRequest):
    try:
        logger.info(f"Received prediction request for {request.symbol}")
        
        # 1. Preprocessing & Data Fetching
        try:
            stock_data, news_data, currency = await data_processor.fetch_data(request.symbol, request.period)
        except Exception as e:
             error_msg = str(e)
             logger.error(f"Error fetching data: {error_msg}")
             
             # Detect network/DNS issues
             if "getaddrinfo failed" in error_msg or "ConnectError" in error_msg:
                 raise HTTPException(status_code=503, detail="Market Data Service Unavailable (Network Connection Error). Please check your internet connection.")
             
             raise HTTPException(status_code=400, detail=f"Failed to fetch data: {error_msg}")

        # 2. AI Inference
        numerical_features, stats = data_processor.prepare_numerical_features(stock_data)
        input_size = numerical_features.shape[1]
        
        # Create sequence for inference
        TIMESTEPS = 60
        if len(numerical_features) >= TIMESTEPS:
             seq = numerical_features[-TIMESTEPS:]
             input_features = seq.reshape(1, TIMESTEPS, -1)
        else:
             padding = np.zeros((TIMESTEPS - len(numerical_features), input_size))
             seq = np.concatenate((padding, numerical_features))
             input_features = seq.reshape(1, TIMESTEPS, input_size)
             
        # Statistical Predictions (passing last_close for denormalization)
        last_close = float(stock_data['close'].iloc[-1])
        
        # Deterministic seed for consistency
        import hashlib
        last_date_str = stock_data['date'].iloc[-1].strftime('%Y-%m-%d')
        seed_val = int(hashlib.md5((request.symbol + last_date_str).encode()).hexdigest(), 16) % (2**32)
        
        # Update predict_price call if needed, or rely on internal predict_forecast changes
        price_forecast = model_inference.predict_price(input_features, last_close, seed=seed_val)
        
        # Sanitize results for JSON
        price_forecast["predicted_price"] = float(np.nan_to_num(price_forecast["predicted_price"], nan=last_close))
        price_forecast["confidence"] = float(np.nan_to_num(price_forecast["confidence"], nan=0.0))
        
        # Sentiment Analysis (FinBERT)
        text_features = data_processor.prepare_text_features(news_data)
        sentiment_score = model_inference.analyze_sentiment(text_features)
        
        if "score" in sentiment_score:
            sentiment_score["score"] = float(np.nan_to_num(sentiment_score["score"], nan=0.0))

        # 3. Decision Logic
        final_decision = decision_engine.make_decision(price_forecast, sentiment_score, stock_data)

        return {
            "symbol": request.symbol,
            "forecast": price_forecast,
            "sentiment": sentiment_score,
            "decision": final_decision,
            "historical_data": stock_data.tail(30).fillna(0).replace([np.inf, -np.inf], 0).to_dict(orient="records")
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks, x_user_id: str = Header(None)):
    try:
        logger.info(f"Starting training request for {request.symbol}")
        
        if not x_user_id:
             raise HTTPException(status_code=401, detail="Unauthorized: Missing User ID")

        if not db_pool:
            raise HTTPException(status_code=500, detail="Database not connected")

        # 1. Create DB Record
        import uuid
        model_id = str(uuid.uuid4()) 
        
        async with db_pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO "Model" (id, name, symbol, type, epochs, status, "createdAt", "userId")
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), $7)
            """, model_id, f"{request.symbol}_Adv_Ensemble", request.symbol, "LSTM+GRU", request.epochs, "TRAINING", x_user_id)

        # 2. Spawn Background Task
        background_tasks.add_task(
            train_model_task, 
            model_id, 
            request.symbol, 
            request.epochs, 
            request.learning_rate, 
            db_pool
        )
        
        return {"status": "success", "message": f"Advanced Training started for {request.symbol}", "model_id": model_id}
    except Exception as e:
        logger.error(f"Training endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/validate/{symbol}")
async def validate_symbol(symbol: str):
    import yfinance as yf
    
    # List of suffixes to try if base fails
    # Empty string '' checks the raw symbol first
    suffixes_to_try = ['', '.NS', '.BO'] 
    
    valid_symbol = None
    currency = "USD"
    
    for suffix in suffixes_to_try:
        candidate = f"{symbol}{suffix}"
        try:
            ticker = yf.Ticker(candidate)
            # Fetch minimal history to verify
            hist = ticker.history(period="1d")
            
            if not hist.empty:
                valid_symbol = candidate
                # Try to capture currency
                if hasattr(ticker, 'fast_info') and 'currency' in ticker.fast_info:
                    currency = ticker.fast_info['currency']
                elif hasattr(ticker, 'info') and 'currency' in ticker.info:
                    currency = ticker.info.get('currency', 'USD')
                break # Found a match, stop searching
        except Exception:
            continue

    if valid_symbol:
        return {"valid": True, "symbol": valid_symbol.upper(), "currency": currency}
    else:
        raise HTTPException(status_code=400, detail=f"Symbol '{symbol}' not found on Yahoo Finance (tried suffixes: .NS, .BO)")

@app.delete("/models/{model_id}")
async def delete_model(model_id: str, x_user_id: str = Header(None)):
    try:
        if not x_user_id:
             raise HTTPException(status_code=401, detail="Unauthorized")
             
        if not db_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
            
        async with db_pool.acquire() as connection:
            # Check ownership
            row = await connection.fetchrow('SELECT "filePath", "userId" FROM "Model" WHERE id = $1', model_id)
            
            if not row:
                 raise HTTPException(status_code=404, detail="Model not found")
                 
            if row['userId'] != x_user_id:
                 raise HTTPException(status_code=403, detail="Forbidden: You do not own this model")
            
            # 1. Get file path to delete file
            if row and row['filePath'] and os.path.exists(row['filePath']):
                try:
                    os.remove(row['filePath'])
                    logger.info(f"Deleted model file: {row['filePath']}")
                except Exception as ex:
                    logger.warning(f"Could not delete file {row['filePath']}: {ex}")

            # 2. Delete from DB
            result = await connection.execute('DELETE FROM "Model" WHERE id = $1', model_id)
            
        return {"status": "success", "message": f"Model {model_id} deleted"}
    except Exception as e:
        logger.error(f"Delete model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_id}/predict")
async def predict_with_model(model_id: str):
    try:
        if not db_pool:
            raise HTTPException(status_code=500, detail="Database not connected")
            
        # 1. Get Model Info
        async with db_pool.acquire() as connection:
            row = await connection.fetchrow('SELECT * FROM "Model" WHERE id = $1', model_id)
            
        if not row:
            raise HTTPException(status_code=404, detail="Model not found")
            
        if row['status'] != 'READY':
             raise HTTPException(status_code=400, detail=f"Model is not ready (Status: {row['status']})")

        # 2. Load Model
        base_path = row['filePath']
        if not base_path or not os.path.exists(f"{base_path}_lstm.pth"):
             logger.warning(f"Model file not found")
             raise HTTPException(status_code=404, detail="Model weights missing. Please retrain.")
        else:
            success = model_inference.load_model(base_path)
            if not success:
                logger.error("Failed to load model weights")
                raise HTTPException(status_code=500, detail="Model file is incompatible. Please retrain this model.")
        
        # 3. Fetch Data
        symbol = row['symbol']
        try:
            stock_data, news_data, currency = await data_processor.fetch_data(symbol, "6mo") 
        except Exception as e:
             error_msg = str(e)
             logger.error(f"Data fetch error for {symbol}: {error_msg}")
             
             if "getaddrinfo failed" in error_msg or "ConnectError" in error_msg:
                 raise HTTPException(status_code=503, detail="Network Error: Cannot reach market data provider. Please check your connectivity.")
                 
             raise HTTPException(status_code=400, detail=f"Invalid Symbol or Data Error: {error_msg}")
        
        # 4. Inference
        # Prepare features (this will now include MA, Vol, Momentum)
        numerical_features, current_stats = data_processor.prepare_numerical_features(stock_data)
        data_raw = numerical_features # Already prepared and scaled using current stats? 
        
        # Actually, we should SCALE using the stats SAVED with the model for consistency
        # But prepare_numerical_features does its own scaling. 
        # Let's re-extract raw and scale with model stats.
        feature_cols = current_stats.get('feature_names', ['close', 'volume', 'ma7_close', 'ma21_close', 'rsi', 'macd', 'macd_signal', 'volatility7', 'returns'])
        
        # Ensure all columns exist
        for col in feature_cols:
             if col not in stock_data.columns:
                  # Fallback logic if some columns missing (shouldn't happen with updated processor)
                  stock_data[col] = 0
                  
        raw_values = stock_data[feature_cols].values
        
        # Normalize using the stats SAVED with the model
        mean_model = model_inference.model_mean
        std_model = model_inference.model_std
        
        # Handle shape mismatch if model was trained on older 2-feature version
        input_size_model = len(mean_model)
        if raw_values.shape[1] != input_size_model:
             logger.warning(f"Feature mismatch: Model expects {input_size_model}, Data has {raw_values.shape[1]}. Adjusting.")
             if raw_values.shape[1] > input_size_model:
                  raw_values = raw_values[:, :input_size_model]
             else:
                  padding = np.zeros((raw_values.shape[0], input_size_model - raw_values.shape[1]))
                  raw_values = np.concatenate((raw_values, padding), axis=1)

        scaled_data = (raw_values - mean_model) / std_model
        
        # Reshape for model (1, 60, input_size)
        TIMESTEPS = 60
        if len(scaled_data) >= TIMESTEPS:
             seq = scaled_data[-TIMESTEPS:]
             input_features = seq.reshape(1, TIMESTEPS, -1)
        else:
             input_size = scaled_data.shape[1]
             padding = np.zeros((TIMESTEPS - len(scaled_data), input_size))
             seq = np.concatenate((padding, scaled_data))
             input_features = seq.reshape(1, TIMESTEPS, input_size)
             
        # 5. Generate 7-Day Forecast
        # Deterministic seed for consistency across tabs/refreshes for the same day
        import hashlib
        import random
        last_date_str = stock_data['date'].iloc[-1].strftime('%Y-%m-%d')
        seed_val = int(hashlib.md5((symbol + last_date_str).encode()).hexdigest(), 16) % (2**32)
        
        forecast_res = model_inference.predict_forecast(input_features, days=7, seed=seed_val)
        norm_returns = forecast_res["forecast_returns"]
        
        if not norm_returns:
             logger.warning("Forecast returned empty.")
             last_close = float(stock_data['close'].iloc[-1])
             forecast_prices = [last_close] * 7
             predicted_price = last_close
             confidence = 0
        else:
            # Reconstruct Prices from Returns
            last_close = float(stock_data['close'].iloc[-1])
            # Historical volatility for noise injection (std of daily returns)
            hist_vol = float(stock_data['returns'].std())
            if np.isnan(hist_vol): hist_vol = 0.01 
            
            forecast_prices = []
            current_price = last_close
            
            # Use local RNG with deterministic seed for consistency
            rng = random.Random(seed_val)
            
            for norm_ret in norm_returns:
                # De-normalize return using model's stats
                denorm_ret = (norm_ret * std_model[0]) + mean_model[0]
                
                # INJECT REALISTIC NOISE: Add small random move based on 10% of hist volatility
                # This prevents "perfectly straight" lines
                noise_pct = rng.uniform(-hist_vol * 0.2, hist_vol * 0.2)
                denorm_ret += noise_pct
                
                # REALISM SAFETY: Cap daily move
                denorm_ret = max(-0.15, min(0.15, denorm_ret))
                
                # Safety against NaN propagation
                if np.isnan(denorm_ret): denorm_ret = 0.0
                
                current_price = current_price * (1 + denorm_ret)
                forecast_prices.append(float(current_price))
            
            # Ensure no NaN/Inf values make it to the JSON response
            forecast_prices = [float(x) if not np.isnan(x) and not np.isinf(x) else float(last_close) for x in forecast_prices]
            predicted_price = forecast_prices[-1]
            confidence = float(np.nan_to_num(forecast_res["confidence"], nan=0.0))
        
        price_forecast = {
            "predicted_price": predicted_price,
            "confidence": confidence
        }
        
        # Sentiment
        text_features = data_processor.prepare_text_features(news_data)
        sentiment_score = model_inference.analyze_sentiment(text_features)
        
        # Sanitize sentiment score for JSON
        if "score" in sentiment_score:
            sentiment_score["score"] = float(np.nan_to_num(sentiment_score["score"], nan=0.0))
        
        decision = decision_engine.make_decision(price_forecast, sentiment_score, stock_data)

        # Generate future dates for the chart (IST)
        import pandas as pd
        from datetime import datetime
        import pytz
        
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist).replace(tzinfo=None)
        
        last_date = stock_data['date'].iloc[-1]
        
        # Calculate future dates starting from the day after last historical data
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
        
        # Metadata about market sync
        days_since_last = (now_ist - last_date).days
        market_status = "OPEN" if days_since_last <= 1 else "CLOSED (Weekend/Holiday)"
        
        forecast_data = [{"date": d.isoformat(), "value": v} for d, v in zip(future_dates, forecast_prices)]

        return {
            "model_id": model_id,
            "symbol": symbol,
            "forecast": price_forecast,
            "sentiment": sentiment_score,
            "decision": decision,
            "historical_data": stock_data.tail(50).fillna(0).replace([np.inf, -np.inf], 0).to_dict(orient="records"),
            "forecast_data": forecast_data,
            "currency": currency,
            "news": news_data,
            "market_info": {
                "last_updated_ist": last_date.isoformat(),
                "current_time_ist": now_ist.isoformat(),
                "status": market_status
            }
        }

    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/news/extract")
async def extract_news(request: dict = Body(...)):
    """
    Extracts the full content of a news article for internal reading.
    Input: {"url": "https://..."}
    """
    try:
        url = request.get("url")
        if not url:
             raise HTTPException(status_code=400, detail="Missing 'url' parameter")
             
        from news_scraper import NewsScraper
        scraper = NewsScraper()
        
        # Scrape content synchronously (since BeautifulSoup is blocking)
        # In prod, run in threadpool
        result = scraper.extract_article_content(url)
        
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
             
        return result
        
    except Exception as e:
        logger.error(f"News extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models(x_user_id: str = Header(None)):
    try:
        if not x_user_id:
             # Requirements: Only see models when signed in.
             return [] 

        if not db_pool:
             return []
             
        async with db_pool.acquire() as connection:
            # Filter by userId
            rows = await connection.fetch('SELECT * FROM "Model" WHERE "userId" = $1 ORDER BY "createdAt" DESC', x_user_id)
            
        # Convert to dict
        models = []
        for row in rows:
            models.append({
                "id": row['id'],
                "name": row['name'],
                "symbol": row['symbol'],
                "type": row['type'],
                "accuracy": f"{row['accuracy']:.1f}%" if row['accuracy'] else "0%",
                "date": row['createdAt'].strftime("%Y-%m-%d %H:%M"),
                "status": row['status'].capitalize()
            })
            
        return models
    except Exception as e:
        logger.error(f"Get models error: {e}")
        return []

@app.get("/models/{model_id}/progress")
async def get_model_progress(model_id: str):
    """
    Returns real-time training progress for a specific model.
    """
    if model_id in training_progress:
        return training_progress[model_id]
    else:
        # Check if model exists in DB to determine if it's pending, failed, or done
        if not db_pool:
             return {"status": "UNKNOWN"}
             
        async with db_pool.acquire() as connection:
            row = await connection.fetchrow('SELECT status FROM "Model" WHERE id = $1', model_id)
            
        if row:
            return {"status": row['status']} # e.g. "READY", "FAILED"
        else:
             raise HTTPException(status_code=404, detail="Model not found")

@app.get("/api/market/analyze")
async def analyze_market(x_user_id: str = Header(None)):
    """
    Fetches real-time data for a predefined universe of stocks,
    calculates risk (volatility) and gain (momentum), and returns structured categories.
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import asyncio
    
    symbols = ['AAPL', 'MSFT', 'NVDA', 'JNJ', 'PG', 'JPM', 'TSLA', 'COIN', 'MSTR', 'DIS', 'PLTR', 'NFLX']
    
    # Run historical download in threadpool
    def fetch_market_data():
        return yf.download(symbols, period="3mo", group_by="ticker", progress=False)
        
    try:
        data = await asyncio.to_thread(fetch_market_data)
        results = []
        
        # We need a small dictionary for company names
        company_names = {
            'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'NVDA': 'NVIDIA Corp.',
            'JNJ': 'Johnson & Johnson', 'PG': 'Procter & Gamble', 'JPM': 'JPMorgan Chase',
            'TSLA': 'Tesla Inc.', 'COIN': 'Coinbase Global', 'MSTR': 'MicroStrategy',
            'DIS': 'Walt Disney', 'PLTR': 'Palantir Tech.', 'NFLX': 'Netflix'
        }
        
        # Parallelize fetching news to reduce latency
        def fetch_ticker_news(sym):
            try:
                ticker_obj = yf.Ticker(sym)
                news = ticker_obj.news
                if news and len(news) > 0 and 'content' in news[0]:
                    title = news[0]['content'].get('title', '')
                    if title:
                        return title
            except Exception:
                pass
            return None
            
        # Optional: Parallel news fetch
        # For performance, we'll fetch them sequentially below or just rely on rule-based analysis if news fails

        for sym in symbols:
            try:
                # Group by ticker makes data[sym] a DataFrame
                df = data[sym] if len(symbols) > 1 else data
                df = df.dropna()
                
                if df.empty or len(df) < 10:
                    continue
                    
                closes = df['Close'].values
                if len(closes) == 0:
                    continue
                    
                # Calculate metrics
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns) * np.sqrt(252)
                gain = (closes[-1] - closes[0]) / closes[0]
                
                # Risk Category
                if volatility < 0.25:
                    risk_level = "low"
                elif volatility < 0.45:
                    risk_level = "medium"
                else:
                    risk_level = "high"
                    
                # Gain Category
                if gain < 0.05:
                    gain_level = "low"
                elif gain < 0.15:
                    gain_level = "medium"
                else:
                    gain_level = "high"
                    
                last_3_days = np.round(closes[-3:], 2).tolist()
                current_price = float(np.round(closes[-1], 2))
                
                quick_analysis = f"Based on 3-month data, {sym} shows {risk_level} volatility ({volatility:.1%}) and {gain_level} return ({gain:.1%})."
                
                news_title = await asyncio.to_thread(fetch_ticker_news, sym)
                if news_title:
                    quick_analysis = f"Recent News: {news_title}. " + quick_analysis

                results.append({
                    "symbol": sym,
                    "name": company_names.get(sym, sym),
                    "riskCategory": risk_level,
                    "gainCategory": gain_level,
                    "currentPrice": current_price,
                    "last3Days": last_3_days,
                    "quickAnalysis": quick_analysis
                })
            except Exception as e:
                logger.error(f"Error processing {sym} in market analysis: {e}")
                
        return {"status": "success", "data": results}
    except Exception as e:
        logger.error(f"Market analysis error: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze market data")

if __name__ == "__main__":
    # Exclude venv and output directories to prevent [WinError 1450] on Windows
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_excludes=["venv", "data", "saved_models", ".git", "__pycache__"]
    )
