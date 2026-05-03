# 🧠 Stocks / Cryprocurrency Price Prediction — Production LSTM Model

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production%20Ready-22C55E?style=flat-square)

**Built by [Teckgeekz](https://teckgeekz.com/ai) · Advanced AI & Machine Learning Engineering**

*End-to-end deep learning pipeline for cryptocurrency or Stocks price forecasting using stacked LSTM networks*

**Inspired by [Stock-Prediction-using-Transformer-NN](https://github.com/Stepka/Stock-Prediction-usning-Transformer-NN)**

**Google Colab [link](https://colab.research.google.com/drive/1wk49JwMQIEc2hAYxV_7UPdh-S_pgIdUI?usp=sharing)**

</div>

---

## 📌 Overview

This repository contains a **production-grade LSTM model** developed by **Teckgeekz** for forecasting SOL-USD (Solana) prices across multiple time horizons. The pipeline spans data ingestion, feature engineering, model training, evaluation, and future forecasting — built with software engineering best practices throughout.

> **Forecast horizons:** 1-day · 3-day · 7-day ahead predictions with 95% confidence intervals

---

## ✨ Key Features

| Capability | Details |
|---|---|
| 📥 **Resilient data ingestion** | yfinance with 3-attempt retry + exponential backoff |
| 🔧 **Feature engineering** | Returns, MA-7, MA-14, 14-day volatility, RSI(14) |
| 🏗️ **Stacked LSTM architecture** | 128 → 64 units with BatchNorm + Dropout regularisation |
| 🛑 **Early stopping** | `restore_best_weights=True`, patience = 20 |
| 📉 **LR scheduling** | ReduceLROnPlateau (factor 0.5, patience 10) |
| 📊 **Multi-metric evaluation** | MAE · RMSE · MAPE · R² |
| 🔮 **Multi-horizon forecasting** | 1, 3, 7 days ahead with 95% CI |
| 📈 **Rich visualisations** | Training curves, actual vs predicted, error fills |

---

## 🏛️ Model Architecture

```
Input: (batch, 14 timesteps, 1 feature)
│
├─ LSTM(128)  →  BatchNorm  →  Dropout(0.2)
├─ LSTM(64)   →  BatchNorm  →  Dropout(0.2)
├─ Dense(32, relu)          →  Dropout(0.1)
├─ Dense(16, relu)
└─ Dense(1)  ← price output
```

**Optimiser:** Adam (lr = 0.001) · **Loss:** MSE · **Epochs:** up to 200 (early stopped) · **Batch size:** 16

---

## 🔬 ML Pipeline

```
Raw OHLCV data (yfinance)
        │
        ▼
  Feature Engineering
  (Returns · MA-7 · MA-14 · Volatility · RSI)
        │
        ▼
  MinMaxScaler  [0, 1]
        │
        ▼
  Train / Test split  (80 / 20)
        │
        ▼
  Sliding window sequences  (lookback = 14)
        │
        ▼
  Stacked LSTM training
  + EarlyStopping + ReduceLROnPlateau
        │
        ▼
  Evaluation  (MAE · RMSE · MAPE · R²)
        │
        ▼
  Future forecasts  (1d · 3d · 7d)
  + 95% Confidence Intervals
```

---

## 📂 Repository Structure

```
sol-lstm-prediction/
│
├── Stock_Prediction_Production.ipynb   # Main training notebook
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
└── outputs/
    ├── training_curves.png             # Loss & MAE plots
    └── predictions_vs_actual.png       # Test set overlay
```

---

## ⚙️ Configuration

All hyperparameters are centralised at the top of the notebook for easy experimentation:

```python
# Asset & data
STOCK          = 'SOL-USD'
LOOKBACK_DAYS  = 1095       # 3 years of history

# Sequence
N_STEPS        = 14         # 2-week lookback window
LOOKUP_STEPS   = [1, 3, 7]  # Forecast horizons

# Training
TRAIN_TEST_SPLIT  = 0.80
BATCH_SIZE        = 16
EPOCHS            = 200
VALIDATION_SPLIT  = 0.15

# Architecture
LSTM_UNITS_1   = 128
LSTM_UNITS_2   = 64
DENSE_UNITS    = 32
DROPOUT_RATE   = 0.2
LEARNING_RATE  = 0.001
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/teckgeekz/lstm-prediction.git
cd lstm-prediction

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Launch Jupyter
jupyter notebook Stock_Prediction_Production.ipynb
```

Or run all cells end-to-end in one shot:

```bash
jupyter nbconvert --to notebook --execute Stock_Prediction_Production.ipynb \
  --output executed_output.ipynb
```

### Requirements

```
yfinance>=0.2.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
tensorflow>=2.13
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
```

---

## 📊 Performance Evaluation

The notebook reports four metrics on the held-out test set:

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | Mean \|actual − predicted\| | Dollar error on average |
| **RMSE** | √Mean(actual − predicted)² | Penalises large errors more heavily |
| **MAPE** | Mean \|actual − predicted\| / actual × 100 | Scale-independent % error |
| **R²** | 1 − SS_res / SS_tot | Variance explained (1.0 = perfect) |

**Quality thresholds used by the model:**

```
MAPE < 5%   → ✅ Excellent  — production ready
MAPE < 10%  → ✅ Good       — use with regular monitoring
MAPE < 20%  → ⚠️  Acceptable — consider architecture changes
MAPE ≥ 20%  → ❌ Poor       — retrain or redesign
```

---

## 🔮 Sample Output

```
======================================================================
FUTURE PRICE PREDICTIONS FOR SOL-USD
======================================================================

Day  1 (2025-05-04):
  Predicted: $148.32
  Range (95% CI): $141.18 - $155.46

Day  3 (2025-05-06):
  Predicted: $151.07
  Range (95% CI): $143.93 - $158.21

Day  7 (2025-05-10):
  Predicted: $155.44
  Range (95% CI): $148.30 - $162.58

======================================================================
Current Price: $146.88
======================================================================
```

---

## 🧩 Technical Highlights

### Stacked LSTM with Regularisation
Two-layer stacked LSTM (128 → 64 units) with BatchNormalisation and Dropout between each block prevents overfitting and stabilises training on the high-variance SOL-USD series.

### Smart Training Callbacks
`EarlyStopping` with `restore_best_weights=True` automatically rolls back to the lowest-validation-loss checkpoint, eliminating the need for manual epoch tuning. `ReduceLROnPlateau` halves the learning rate when progress stalls, enabling fine convergence.

### Uncertainty Quantification
Future predictions include 95% confidence intervals derived from the standard deviation of test-set residuals, giving a principled uncertainty band rather than a single point estimate.

### Production-Grade Data Ingestion
Exponential backoff retry logic handles transient yfinance API failures, and explicit MultiIndex column handling ensures compatibility with the new yfinance ≥ 0.2.x API structure.

---

## 🗺️ Roadmap

- [ ] Wire engineered features (RSI, volatility, MAs) into LSTM input as multi-channel time series
- [ ] Implement iterative multi-step forecasting for 3- and 7-day horizons
- [ ] Add walk-forward cross-validation for more robust metric estimates
- [ ] Persist model with `model.save()` and scaler with `joblib.dump()`
- [ ] Experiment with additional features: volume, on-chain metrics, BTC correlation
- [ ] Containerise with Docker for reproducible deployment
- [ ] REST API endpoint for real-time inference (FastAPI)

---

## ⚠️ Disclaimer

This project is built for **research and educational purposes only**. Price predictions from any machine learning model carry inherent uncertainty and should **not** be used as the sole basis for financial decisions. Always apply risk management when trading.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---


**Made with precision by [Teckgeekz](https:teckgeekz.com/ai)**

*Building production-grade AI systems · Deep Learning · Time Series · Quantitative Finance*

⭐ Star this repo if you found it useful



[def]: https://github.com/Stepka/Stock-Prediction-usning-Transformer-NN
