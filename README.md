# 📈 Stock Price Prediction with TCN-BiLSTM

A hybrid deep learning model that combines **Temporal Convolutional Networks (TCN)** and **Bidirectional LSTM (BiLSTM)** for accurate stock price forecasting. This project covers the full pipeline: data collection and preprocessing, hyperparameter tuning, model training, and prediction evaluation.

---

## 🗂️ Repository Structure

```
Stock-Price-Prediction-TCN-BiLSTM/
│
├── data/                          # Raw and processed stock data
├── prediction/                    # Output prediction results & plots
├── saved_models/                  # Trained model checkpoints
├── my_tuner_dir/
│   └── stock_price_prediction_hyperband/  # Keras Tuner hyperband search results
│
├── Data.ipynb                     # Data collection, EDA, and preprocessing
├── model.ipynb                    # Model architecture, training, and hyperparameter tuning
├── test.ipynb                     # Model evaluation and prediction visualization
└── data.zip                       # Compressed dataset archive
```

---

## 🧠 Model Architecture

This project uses a **hybrid TCN-BiLSTM** architecture designed to leverage the strengths of both network types for time-series prediction:

| Component | Role |
|---|---|
| **TCN (Temporal Convolutional Network)** | Captures short-to-medium-term local patterns via causal dilated convolutions |
| **BiLSTM (Bidirectional LSTM)** | Captures long-range bidirectional temporal dependencies in the sequence |
| **Hyperband Tuner** | Automated hyperparameter optimization via Keras Tuner |

**Why this hybrid?**
- TCN provides efficient, parallelizable convolution over long sequences with its dilated causal architecture — it avoids the vanishing gradient problem common in RNNs.
- BiLSTM processes sequences in both forward and backward directions, capturing context from both past and future time steps.
- Together, the two networks complement each other: TCN handles local feature extraction while BiLSTM models global sequence dependencies.

---

## 📓 Notebooks

### `Data.ipynb`
Handles all data-related steps:
- Fetching historical stock price data
- Exploratory Data Analysis (EDA)
- Feature engineering and technical indicator computation
- Data normalization and sequence windowing for time-series input

### `model.ipynb`
Covers model construction and training:
- Building the TCN-BiLSTM hybrid model in Keras/TensorFlow
- Hyperparameter search using **Keras Tuner with Hyperband** strategy
- Training the best model configuration
- Saving trained model weights to `saved_models/`

### `test.ipynb`
Handles evaluation and visualization:
- Loading the saved model
- Running predictions on the test set
- Comparing predicted vs. actual stock prices
- Plotting prediction results (saved to `prediction/`)

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install tensorflow keras keras-tuner numpy pandas matplotlib scikit-learn yfinance
```

### Clone the Repository

```bash
git clone https://github.com/Poor1ST/Stock-Price-Prediction-TCN-BiLSTM.git
cd Stock-Price-Prediction-TCN-BiLSTM
```

### Extract the Data

```bash
unzip data.zip -d data/
```

### Run the Notebooks in Order

1. **Data preparation** → Open and run `Data.ipynb`
2. **Model training** → Open and run `model.ipynb`
3. **Evaluation** → Open and run `test.ipynb`

> It is recommended to run these notebooks sequentially, as each step depends on outputs from the previous one.

---

## 🔧 Hyperparameter Tuning

This project uses **Keras Tuner with the Hyperband** algorithm to efficiently search the hyperparameter space. Tuning results are saved in:

```
my_tuner_dir/stock_price_prediction_hyperband/
```

Tunable parameters typically include:
- Number of TCN filters and kernel size
- Number of BiLSTM units
- Dropout rate
- Learning rate
- Number of stacked layers

The best configuration found during the search is automatically selected and used for the final training run.

---

## 📊 Results

Prediction outputs and comparison plots are saved in the `prediction/` directory after running `test.ipynb`. The model is evaluated using standard regression metrics such as:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

---

## 🛠️ Tech Stack

- **Python** — Core language
- **TensorFlow / Keras** — Deep learning framework
- **Keras Tuner** — Automated hyperparameter search (Hyperband)
- **NumPy / Pandas** — Data manipulation
- **Matplotlib** — Visualization
- **scikit-learn** — Preprocessing and metrics
- **yfinance** *(or similar)* — Stock data retrieval

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Stock price predictions made by this model should **not** be used as the basis for real financial decisions. Financial markets are inherently unpredictable and no model can guarantee future returns.

---

## 📄 License

This project is open-source. Feel free to fork, modify, and build upon it.

---

## 🙌 Acknowledgements

- [Keras Tuner](https://keras.io/keras_tuner/) for automated hyperparameter optimization
- The broader research community on hybrid deep learning models for financial time-series forecasting
