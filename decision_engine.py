from typing import Dict, Any

class DecisionEngine:
    def __init__(self):
        pass

    def make_decision(self, price_forecast: Dict[str, float], sentiment: Dict[str, Any], historical_data: Any) -> Dict[str, Any]:
        """
        Combines Math (Price models) and Sentiment (FinBERT) to give a final recommendation.
        """
        price = price_forecast["predicted_price"]
        sentiment_score = sentiment["score"]
        sentiment_label = sentiment["label"]
        
        current_price = historical_data.iloc[-1]['close']
        
        # Calculate percentage change
        pct_change = ((price - current_price) / current_price) * 100
        
        # Determine Signal
        action = "HOLD"
        
        # Sensitivity: Reduced threshold from 2% to 1% for more active signals
        # We also use the raw sentiment_score for more nuance
        if pct_change > 1:
            if sentiment_score > 0.3:
                action = "STRONG BUY"
            elif sentiment_score < -0.3:
                action = "HOLD" # Divergence
            else:
                action = "BUY"
        elif pct_change < -1:
            if sentiment_score < -0.3:
                action = "STRONG SELL"
            elif sentiment_score > 0.3:
                action = "HOLD" # Divergence
            else:
                action = "SELL"
        else:
            # Even if price is stable, strong sentiment can move the needle to BUY/SELL
            if sentiment_score > 0.5:
                action = "BUY (Sentiment-Led)"
            elif sentiment_score < -0.5:
                action = "SELL (Sentiment-Led)"
        
        # Generate Detailed Analysis Summary
        trend_desc = "upward" if pct_change > 0 else "downward"
        
        description = (
            f"The AI model predicts a {trend_desc} trend of approximately {pct_change:.2f}% over the 7-day forecast. "
            f"Market sentiment is currently {sentiment_label.upper()} with a score of {sentiment_score:.2f}. "
        )
        
        if action == "STRONG BUY":
            description += "Technical models and news sentiment are both strongly positive, indicating a high-conviction buy opportunity."
        elif action == "STRONG SELL":
            description += "Technical and sentiment indicators are aggressively negative. Strongly recommend an immediate SELL."
        elif action == "HOLD":
            description += "Price forecasts and recent news show strictly conflicting signals. Firmly HOLD positions until clarity emerges."
        elif "Sentiment-Led" in action:
             description += "While technical price movement is sideways, heavy social/news sentiment is driving this recommendation."
        else:
            description += f"A moderate {action.lower()} is suggested as the market shows clear directional bias."

        return {
            "recommendation": action,
            "signal_strength": "HIGH" if abs(pct_change) > 4 or abs(sentiment_score) > 0.6 else "MEDIUM" if abs(pct_change) > 1 else "LOW",
            "reasoning": description
        }
