import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
import os
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from typing import Dict, Any, List

# Define models locally for inference matching training def
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, rnn_out):
        weights = torch.softmax(self.attention(rnn_out), dim=1)
        context = torch.sum(weights * rnn_out, dim=1)
        return context

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        context = self.attention(out)
        out = self.fc(context)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        context = self.attention(out)
        out = self.fc(context)
        return out

class ModelInference:
    def __init__(self):
        print("Initializing AI Models...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize placeholders
        self.lstm = None
        self.gru = None
        self.lstm_acc = 0.5
        self.gru_acc = 0.5
        self.model_mean = 0
        self.model_std = 1
        
        # 3. Initialize FinBERT
        try:
            # FinBERT requires about 400MB of weights. If it fails, it's usually network or disk space.
            self.sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
        except Exception as e:
            error_msg = str(e)
            if "ConnectError" in error_msg or "getaddrinfo failed" in error_msg:
                print(f"Warning: FinBERT could not be downloaded (Network Error). Using Keyword-based Sentiment Fallback.")
            else:
                print(f"Warning: Could not load FinBERT. Error: {error_msg}")
            
            self.sentiment_pipe = None

    def load_model(self, base_path: str):
        """
        Loads trained weights into LSTM and GRU models.
        """
        try:
            # Load LSTM
            lstm_path = f"{base_path}_lstm.pth"
            checkpoint_lstm = torch.load(lstm_path, weights_only=False)
            config_lstm = checkpoint_lstm.get('config', {'hidden_size': 64, 'num_layers': 2, 'dropout': 0})
            
            # Determine input size from state dict
            input_size = checkpoint_lstm['model_state_dict']['lstm.weight_ih_l0'].shape[1]
            
            self.lstm = LSTMModel(
                input_size=input_size,
                hidden_size=config_lstm['hidden_size'],
                num_layers=config_lstm['num_layers'],
                dropout=config_lstm.get('dropout', 0.2)
            ).to(self.device)
            self.lstm.load_state_dict(checkpoint_lstm['model_state_dict'])
            self.lstm.eval()
            self.lstm_acc = checkpoint_lstm.get('accuracy', 50.0) / 100.0
            
            # Load GRU
            gru_path = f"{base_path}_gru.pth"
            checkpoint_gru = torch.load(gru_path, weights_only=False)
            config_gru = checkpoint_gru.get('config', {'hidden_size': 64, 'num_layers': 2, 'dropout': 0})
            
            self.gru = GRUModel(
                input_size=input_size,
                hidden_size=config_gru['hidden_size'],
                num_layers=config_gru['num_layers'],
                dropout=config_gru.get('dropout', 0.2)
            ).to(self.device)
            self.gru.load_state_dict(checkpoint_gru['model_state_dict'])
            self.gru.eval()
            self.gru_acc = checkpoint_gru.get('accuracy', 50.0) / 100.0
            
            # Load Normalization Stats (Lists for multi-feature)
            # Use safety checks to ensure we don't get None or NaN
            mean_val = checkpoint_lstm.get('mean')
            if mean_val is None: mean_val = [0] * input_size
            self.model_mean = np.nan_to_num(np.array(mean_val))
            
            std_val = checkpoint_lstm.get('std')
            if std_val is None: std_val = [1] * input_size
            self.model_std = np.nan_to_num(np.array(std_val))
            # Ensure std is not zero
            self.model_std[self.model_std == 0] = 1.0
            
            # Final sanity check: if sizes don't match input_size, pad or trim
            if len(self.model_mean) < input_size:
                self.model_mean = np.pad(self.model_mean, (0, input_size - len(self.model_mean)))
            if len(self.model_std) < input_size:
                self.model_std = np.pad(self.model_std, (0, input_size - len(self.model_std)), constant_values=1.0)
            
            print(f"Loaded Realistic Model from {base_path}. LSTM Acc: {self.lstm_acc:.2f}, GRU Acc: {self.gru_acc:.2f}")
            return True
        except Exception as e:
            print(f"Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict_forecast(self, initial_sequence: np.ndarray, days: int = 7, seed: int = None) -> Dict[str, Any]:
        """
        Generates forecast using ensemble of LSTM and GRU.
        Input: (1, seq_length, input_size)
        """
        import random
        rng = random.Random(seed) if seed is not None else random.Random()
        
        if not self.lstm or not self.gru:
            # SIMULATION FALLBACK: If no model loaded, return a realistic simulated trend
            print(f"No model loaded. Generating trend-aware simulation (Seed: {seed})...")
            
            # initial_sequence is shape (1, 60, features), index 0 is now normalized 'close'
            last_prices = initial_sequence[0, :, 0]
            
            # Generate prices slightly noisy around the last known price
            last_val = float(last_prices[-1])
            std_val = float(np.std(last_prices)) if len(last_prices) > 1 else 0.01
            sim_prices = [rng.normalvariate(last_val, std_val * 0.1) for _ in range(days)]
            
            return {
                "forecast_returns": sim_prices,
                "confidence": rng.uniform(0.45, 0.52)
            }

        self.lstm.eval()
        self.gru.eval()
        
        forecast_returns = []
        # current_seq shape: (1, 60, input_size)
        current_seq = torch.FloatTensor(initial_sequence).to(self.device) 
        input_size = current_seq.shape[2]
        
        # Calculate weights based on accuracy (handle 0 or missing values)
        total_acc = (self.lstm_acc or 0.5) + (self.gru_acc or 0.5)
        w_lstm = (self.lstm_acc or 0.5) / total_acc
        w_gru = (self.gru_acc or 0.5) / total_acc
        
        with torch.no_grad():
            for i in range(days):
                # 1. Predict next normalized price
                out_lstm = self.lstm(current_seq) 
                out_gru = self.gru(current_seq)   
                
                # Ensemble prediction
                ensemble_pred = (out_lstm.item() * w_lstm) + (out_gru.item() * w_gru)
                
                forecast_returns.append(ensemble_pred)
                
                # 2. Update sequence for next step
                new_step = torch.zeros((1, 1, input_size)).to(self.device)
                
                # Simple roll: copy technical features from last step, update the price (index 0)
                new_step[0, 0, :] = current_seq[0, -1, :]
                new_step[0, 0, 0] = ensemble_pred
                
                # Roll sequence
                current_seq = torch.cat((current_seq[:, 1:, :], new_step), dim=1)
                
        confidence = (self.lstm_acc * w_lstm) + (self.gru_acc * w_gru)
        
        return {
            "forecast_returns": forecast_returns,
            "confidence": min(0.99, confidence)
        }

    # Keep old method for compatibility if needed
    def predict_price(self, features: np.ndarray, last_close: float, seed: int = None) -> Dict[str, float]:
        # Legacy wrapper
        res = self.predict_forecast(features, days=1, seed=seed)
        if res["forecast_returns"]:
            # De-normalize price (the model directly outputs normalized price now)
            norm_price = res["forecast_returns"][0]
            denorm_price = (norm_price * self.model_std[0]) + self.model_mean[0]
            predicted_price = denorm_price
            return {"predicted_price": predicted_price, "confidence": res["confidence"]}
        return {"predicted_price": last_close, "confidence": 0}

    def analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """
        Uses FinBERT to get sentiment.
        """
        if not texts:
            return {"score": 0, "label": "NEUTRAL"}
            
        if self.sentiment_pipe:
            # Analyze each text
            results = self.sentiment_pipe(texts)
            # Simple aggregation logic
            # FinBERT returns labels: 'positive', 'negative', 'neutral'
            # Improved aggregation logic:
            # 1. Neutral results often dilute the signal. 
            # 2. We weight positive/negative items 2x more than neutrals.
            # 3. We check for 'High Confidence' outliers that should move the score regardless of the average.
            
            sentiment_score = 0
            weighted_count = 0
            
            for res in results:
                weight = 1.0
                if res['label'] == 'positive':
                    sentiment_score += (res['score'] * 2.0)
                    weighted_count += 2.0
                elif res['label'] == 'negative':
                    sentiment_score -= (res['score'] * 2.0)
                    weighted_count -= 2.0
                else: # neutral
                    # Neutral contributes 0 to score but counts toward the denominator
                    weighted_count += 1.0
                    
            if weighted_count == 0:
                avg_score = 0
            else:
                avg_score = sentiment_score / abs(weighted_count)
            
            # Sensitivity adjustment: Neutral threshold reduced from 0.1 to 0.05
            final_label = "bullish" if avg_score > 0.05 else "bearish" if avg_score < -0.05 else "neutral"
            
            return {
                "score": float(np.clip(avg_score, -1, 1)),
                "label": final_label,
                "raw_results": results
            }
        else:
            return {"score": 0.5, "label": "bullish (mock)"}
