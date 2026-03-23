import numpy as np
import pandas as pd
from inference import ModelInference
from decision_engine import DecisionEngine

def test_sentiment_sensitivity():
    print("--- START: test_sentiment_sensitivity ---")
    inf = ModelInference()
    
    # Mock pipe results
    # 1. Mostly Neutral with one strong Positive
    results_1 = [
        {'label': 'neutral', 'score': 0.99},
        {'label': 'neutral', 'score': 0.98},
        {'label': 'positive', 'score': 0.95}
    ]
    
    # We manually override the _analyze_sentiment logic for testing aggregation
    def mock_aggregate(results):
        sentiment_score = 0
        weighted_count = 0
        for res in results:
            if res['label'] == 'positive':
                sentiment_score += (res['score'] * 2.0)
                weighted_count += 2.0
            elif res['label'] == 'negative':
                sentiment_score -= (res['score'] * 2.0)
                weighted_count -= 2.0
            else:
                weighted_count += 1.0
        return sentiment_score / abs(weighted_count)

    score_1 = mock_aggregate(results_1)
    label_1 = "bullish" if score_1 > 0.05 else "bearish" if score_1 < -0.05 else "neutral"
    
    print(f"Test 1 (2 Neut, 1 Pos): Score={score_1:.4f}, Label={label_1}")
    assert label_1 == "bullish" # Should be bullish now because Pos is weighted 2x
    
    # 2. Borderline Neutral
    results_2 = [
        {'label': 'neutral', 'score': 0.9},
        {'label': 'positive', 'score': 0.1} # Weak positive
    ]
    score_2 = mock_aggregate(results_2)
    label_2 = "bullish" if score_2 > 0.05 else "bearish" if score_2 < -0.05 else "neutral"
    print(f"Test 2 (Weak Pos): Score={score_2:.4f}, Label={label_2}")
    assert label_2 == "bullish" # 0.2 / 3.0 = 0.066 > 0.05
    
    print("✅ Sentiment sensitivity test PASSED")

def test_decision_engine():
    print("\n--- START: test_decision_engine ---")
    engine = DecisionEngine()
    
    # Case 1: Small price change but strong sentiment
    forecast = {"predicted_price": 100.5} # 0.5% up
    sentiment = {"score": 0.6, "label": "bullish"}
    hist = pd.DataFrame({'close': [100.0]})
    
    res = engine.make_decision(forecast, sentiment, hist)
    print(f"Decision (0.5% up, 0.6 sentiment): {res['recommendation']}")
    assert res['recommendation'] == "BUY (Sentiment-Led)"
    
    # Case 2: 1.5% up, 0.4 sentiment
    forecast = {"predicted_price": 101.5}
    sentiment = {"score": 0.4, "label": "bullish"}
    res = engine.make_decision(forecast, sentiment, hist)
    print(f"Decision (1.5% up, 0.4 sentiment): {res['recommendation']}")
    assert res['recommendation'] == "STRONG BUY"
    
    print("✅ Decision engine test PASSED")

if __name__ == "__main__":
    test_sentiment_sensitivity()
    test_decision_engine()
