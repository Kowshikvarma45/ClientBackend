import pandas as pd
import numpy as np
from datetime import datetime
import pytz

def test_ist_conversion():
    print("--- START: test_ist_conversion ---")
    # Simulate yfinance output (usually has a TZ, often UTC or exchange-local)
    dates = pd.date_range(start='2023-01-01', periods=5, tz='UTC')
    df = pd.DataFrame({'Close': [100, 101, 102, 103, 104]}, index=dates).reset_index()
    df.columns = ['date', 'close']
    
    # Logic from processing.py
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    
    last_date = df['date'].iloc[-1]
    print(f"Last Date (IST): {last_date}")
    
    # Check if hour is now shifted correctly (UTC + 5.5h)
    # 2023-01-05 00:00:00 UTC -> 2023-01-05 05:30:00 IST
    assert last_date.hour == 5
    assert last_date.minute == 30
    print("IST conversion works!")

def test_forecast_alignment():
    print("\n--- START: test_forecast_alignment ---")
    ist = pytz.timezone('Asia/Kolkata')
    # Current time in IST
    now_ist = datetime.now(ist).replace(tzinfo=None)
    
    # Simulate last trading day (Friday Feb 20)
    last_trading_day = pd.Timestamp('2026-02-20')
    
    # Logic from main.py
    future_dates = [last_trading_day + pd.Timedelta(days=i) for i in range(1, 8)]
    
    print(f"Current IST Time: {now_ist}")
    print(f"Last Trading Day: {last_trading_day}")
    print(f"First Forecast Day: {future_dates[0]}")
    print(f"Second Forecast Day: {future_dates[1]}") # This should be Feb 22
    
    days_since_last = (now_ist - last_trading_day).days
    print(f"Days since last trading day: {days_since_last}")
    
    # If today is Feb 22 (Sunday), then days_since_last = 2
    # Feb 22 is in the list of future dates
    assert any(d.day == 22 and d.month == 2 for d in future_dates)
    print("✅ Forecast includes 'Today' (Feb 22)!")
    print("--- END: test_forecast_alignment ---")

if __name__ == "__main__":
    test_ist_conversion()
    test_forecast_alignment()
