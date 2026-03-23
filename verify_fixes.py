import asyncio
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from processing import DataProcessor
import json

async def test_json_compliance():
    print("\n--- START: test_json_compliance ---")
    dp = DataProcessor(api_key="test")
    mock_ticker = MagicMock()
    mock_history = pd.DataFrame({
        'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Volume': [1000, 0, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }, index=pd.date_range(start='2023-01-01', periods=11))
    
    with patch('asyncio.get_event_loop') as mock_loop_getter:
        mock_loop = MagicMock()
        mock_loop_getter.return_value = mock_loop
        mock_loop.run_in_executor = AsyncMock()
        mock_loop.run_in_executor.side_effect = [mock_ticker, mock_history]
        
        try:
            df, news, currency = await dp.fetch_data("AAPL", "1mo")
            data = df.tail(30).fillna(0).replace([np.inf, -np.inf], 0).to_dict(orient="records")
            
            sanitized_data = []
            for record in data:
                sanitized_record = {}
                for k, v in record.items():
                    if isinstance(v, (np.float64, float, np.int64, int)):
                        sanitized_record[k] = float(v) if isinstance(v, (np.float64, float)) else int(v)
                    else:
                        sanitized_record[k] = str(v)
                sanitized_data.append(sanitized_record)

            json_str = json.dumps(sanitized_data)
            print("JSON dump successful")
            print("✅ test_json_compliance PASSED")
        except Exception as e:
            print(f"❌ test_json_compliance FAILED: {e}")
    print("--- END: test_json_compliance ---")

async def test_feature_expansion():
    print("\n--- START: test_feature_expansion ---")
    dp = DataProcessor(api_key="test")
    mock_ticker = MagicMock()
    # Need at least 20 days for MOMENTUM (MA20)
    dates = pd.date_range(start='2023-01-01', periods=30)
    mock_history = pd.DataFrame({
        'Close': np.linspace(100, 110, 30),
        'Volume': np.linspace(1000, 1500, 30)
    }, index=dates)
    
    with patch('asyncio.get_event_loop') as mock_loop_getter:
        mock_loop = MagicMock()
        mock_loop_getter.return_value = mock_loop
        mock_loop.run_in_executor = AsyncMock()
        mock_loop.run_in_executor.side_effect = [mock_ticker, mock_history]
        
        try:
            df, news, currency = await dp.fetch_data("AAPL", "1mo")
            scaled_data, stats = dp.prepare_numerical_features(df)
            
            print(f"Feature count: {scaled_data.shape[1]}")
            print(f"Feature names: {stats.get('feature_names')}")
            
            assert scaled_data.shape[1] == 5
            assert 'ma7_return' in stats['feature_names']
            assert 'momentum' in stats['feature_names']
            print("✅ test_feature_expansion PASSED")
        except Exception as e:
            print(f"❌ test_feature_expansion FAILED: {e}")
            import traceback
            traceback.print_exc()
    print("--- END: test_feature_expansion ---")

if __name__ == "__main__":
    asyncio.run(test_json_compliance())
    asyncio.run(test_feature_expansion())
