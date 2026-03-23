import yfinance as yf
import pandas as pd
import numpy as np

def analyze_market():
    symbols = ['AAPL', 'MSFT', 'NVDA', 'JNJ', 'PG', 'JPM', 'TSLA', 'COIN', 'MSTR', 'DIS']
    
    # Download last 3 months
    data = yf.download(symbols, period="3mo", group_by="ticker", progress=False)
    
    results = []
    
    for sym in symbols:
        try:
            # Handle Single vs Multi Index differences
            df = data[sym] if len(symbols) > 1 else data
            
            # Drop NaN rows
            df = df.dropna()
            if df.empty or len(df) < 10:
                continue
                
            closes = df['Close'].values
            
            # Simple daily returns
            returns = np.diff(closes) / closes[:-1]
            
            # Volatility (Risk) - annualized approx
            volatility = np.std(returns) * np.sqrt(252)
            
            # Gain Potential (3-month return)
            gain = (closes[-1] - closes[0]) / closes[0]
            
            # Categorize Risk
            if volatility < 0.20:
                risk_level = "low"
            elif volatility < 0.40:
                risk_level = "medium"
            else:
                risk_level = "high"
                
            # Categorize Gain
            if gain < 0.05:
                gain_level = "low"
            elif gain < 0.15:
                gain_level = "medium"
            else:
                gain_level = "high"
                
            # Recent prices
            last_3_days = np.round(closes[-3:], 2).tolist()
            current_price = np.round(closes[-1], 2)
            
            # Try to get news
            quick_analysis = f"Based on 3-month data, {sym} shows {risk_level} volatility ({volatility:.1%}) and {gain_level} momentum ({gain:.1%})."
            
            ticker_obj = yf.Ticker(sym)
            try:
                news = ticker_obj.news
                if news and len(news) > 0 and 'content' in news[0]:
                    title = news[0]['content'].get('title', '')
                    if title:
                        quick_analysis = f"Recent News: {title}. " + quick_analysis
            except Exception as e:
                pass
            
            results.append({
                "symbol": sym,
                "name": sym, # Could fetch from info but it's slow
                "riskCategory": risk_level,
                "gainCategory": gain_level,
                "currentPrice": current_price,
                "last3Days": last_3_days,
                "quickAnalysis": quick_analysis
            })
        except Exception as e:
            print(f"Error on {sym}: {e}")
            
    return results

if __name__ == "__main__":
    res = analyze_market()
    for r in res:
        print(f"{r['symbol']} | {r['riskCategory'].upper()} Risk | {r['gainCategory'].upper()} Gain | ${r['currentPrice']}")
        print(f"  Prices: {r['last3Days']}")
        print(f"  Analysis: {r['quickAnalysis']}")
        print()
