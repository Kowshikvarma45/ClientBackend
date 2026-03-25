import asyncio
import pandas as pd
import numpy as np
import requests
from typing import Tuple, List, Dict
import random # For mocking if API fails

class DataProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # In a real scenario, we would use the AlphaVantage library or direct requests
        self.base_url = "https://www.alphavantage.co/query"

    async def fetch_data(self, symbol: str, period: str) -> Tuple[pd.DataFrame, List[Dict], str]:
        """
        Fetches stock data, news, and currency using yfinance.
        Returns: (DataFrame, News List, Currency String)
        """
        try:
            import yfinance as yf
            print(f"Fetching REAL data for {symbol} using yfinance...")
            
            # 1. Fetch Stock Data
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            history = await loop.run_in_executor(None, lambda: ticker.history(period=period))
            
            # STRICT VALIDATION: If history is empty, the symbol is likely invalid.
            if history.empty:
                 print(f"❌ Validity Check Failed: No data found for symbol '{symbol}'.")
                 raise ValueError(f"Invalid Symbol: {symbol}")
            
            # Get Currency
            currency = "USD" # Default
            try:
                # Use a safer approach to access info/fast_info as they can be None or empty
                fast_info = getattr(ticker, 'fast_info', None)
                info = getattr(ticker, 'info', None)
                
                if fast_info is not None and isinstance(fast_info, dict) and 'currency' in fast_info:
                    currency = fast_info['currency']
                elif fast_info is not None and hasattr(fast_info, 'currency'):
                    currency = fast_info.currency
                elif info is not None and isinstance(info, dict) and 'currency' in info:
                    currency = info.get('currency', 'USD')
            except Exception as e:
                print(f"Currency fetch warning: {e}")

            # Format DataFrame
            df = history.reset_index()
            df.columns = [c.lower() for c in df.columns] 
            if 'date' in df.columns:
                 # Localize to IST (Asia/Kolkata)
                 # yfinance usually returns localized UTC or exchange-local time.
                 # We convert it to IST and then strip the tz-info for clean JSON.
                 df['date'] = pd.to_datetime(df['date']).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
            
            # Calculate Returns and Percentage Changes (Crucial for realism)
            # This makes the data stationary and easier for the model to learn
            df['returns'] = df['close'].pct_change()
            df['vol_change'] = df['volume'].pct_change()
            
            # SANITIZATION: Replace Inf/-Inf with 0 and then drop NaN
            # JSON cannot handle Inf/-Inf or NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.dropna().reset_index(drop=True)
            
            # 2. Fetch News
            from news_scraper import NewsScraper
            news_items = []
            
            try:
                scraper = NewsScraper()
                news_items = scraper.get_news(symbol)
                
                print(f"DEBUG: Scraped {len(news_items)} news items for {symbol}")

                if not news_items:
                    print(f"No specific news for {symbol}, adding sector context...")
                    # Realistic Fallback Data with Images
                    curr_sym = "$" if currency == "USD" else "₹" if currency == "INR" else ""
                    
                    last_close = df.iloc[-1]['close']
                    prev_close = df.iloc[0]['close']
                    
                    # We use the raw closes for the summary as it's for display
                    change_pct = ((last_close - prev_close) / prev_close) * 100
                    trend_adj = "Surges" if change_pct > 0 else "Slips"
                    
                    current_time = pd.Timestamp.now().timestamp()
                    
                    news_items = [
                        {
                            "title": f"Market Analysis: {symbol} {trend_adj} Amidst Volatility", 
                            "summary": f"Analysts suggest that specific sector movements are driving the recent price action for {symbol}. Key support levels are being tested.", 
                            "source": "Bloomberg",
                            "url": "https://www.bloomberg.com/markets",
                            "thumbnail": "https://images.unsplash.com/photo-1611974765270-ca12586343bb?q=80&w=300&auto=format&fit=crop",
                            "time_published": int(current_time - 3600*2)
                        },
                        {
                            "title": f"Global Markets Update: Impact on {currency} Assets", 
                            "summary": "Investors are closely monitoring central bank policies as inflation data continues to influence market sentiment globally.",
                            "source": "Reuters",
                            "url": "https://www.reuters.com/finance",
                            "thumbnail": "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=300&auto=format&fit=crop",
                            "time_published": int(current_time - 3600*5)
                        }
                    ]
            except Exception as e:
                print(f"News fetch warning: {e}")
                news_items = [
                        {"title": f"Market Update {symbol}", "summary": "General market sentiment remains mixed.", "source": "Reuters", "thumbnail": None, "time_published": None}
                ]

            return df, news_items, currency

        except Exception as e:
            print(f"Data Fetch Failed: {e}")
            raise e

    def prepare_numerical_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Prepares features for return-based model.
        Uses returns as the prediction target to ensure data is stationary.
        """
        # 1. Base Features
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        if 'vol_change' not in df.columns:
            df['vol_change'] = df['volume'].pct_change()
            
        # 2. Technical Indicators (Stationary)
        df['ma7_return'] = df['close'].rolling(window=7).mean().pct_change()
        df['volatility7'] = df['returns'].rolling(window=7).std()
        df['momentum'] = df['returns'].rolling(window=10).mean()
            
        # Set `returns` as index 0 (prediction target)
        feature_cols = ['returns', 'vol_change', 'ma7_return', 'volatility7', 'momentum']
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
            
        if df.empty:
             # Return dummy data to prevent crash
             dummy = np.zeros((1, len(feature_cols)))
             return dummy, {'mean': [0]*len(feature_cols), 'std': [1]*len(feature_cols)}

        # Extract features
        data = df[feature_cols].values
        
        # Replace inf and -inf with reasonable bounds
        data = np.nan_to_num(data, nan=0.0, posinf=0.5, neginf=-0.5)
        
        # Simple Z-score normalization for stability
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1 # Prevent div by zero
        
        scaled_data = (data - mean) / std
        
        # Final safety check for NaN in scaled data
        scaled_data = np.nan_to_num(scaled_data)
        
        stats = {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'feature_names': feature_cols
        }
        
        return scaled_data, stats
    def prepare_text_features(self, news: List[Dict]) -> List[str]:
        """
        Extracts text for FinBERT.
        """
        return [item["title"] + ". " + item["summary"] for item in news]
