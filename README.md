Made to practice might really work

1. Model Architecture
Input Layer:

The input to the model consists of sequences of features that include the Bitcoin closing prices and several technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV).
The sequence length is set to 240, meaning that the model uses the last 240 days of data to make predictions for the next day.
LSTM Layers:

The model contains 3 LSTM layers with 150 hidden units in each layer.
The LSTM layers process the sequence of inputs and retain information over time, making them well-suited for time series data.
LSTMs have internal memory cells that allow them to "remember" long-term dependencies in the data, which is essential for capturing market trends over time.
Dropout Regularization:

A dropout layer with a rate of 0.4 is applied after each LSTM layer to prevent overfitting. Dropout randomly sets a fraction of the units to zero during training, which helps the model generalize better to unseen data.
Fully Connected (Linear) Layer:

After the final LSTM layer, the output is passed through a fully connected (linear) layer. This layer maps the LSTM’s output to a single value, which is the predicted Bitcoin price for the next day.
Activation Functions:

The model doesn’t use an activation function in the output layer because this is a regression problem (predicting continuous values like Bitcoin prices), where a linear output is more appropriate.
2. Model Input
Features: The input data includes 8 features:

Price (closing price of Bitcoin)
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
MACD Signal
MACD Histogram
Bollinger Upper Band
Bollinger Lower Band
ATR (Average True Range)
The input is fed into the LSTM layers in sequences of 240 days, which means each sequence has a shape of (240, 8).

3. Training Process
Loss Function: The model uses Mean Squared Error (MSE) as the loss function, which measures the difference between the predicted and actual prices. Lower MSE indicates better predictions.

Optimizer: The AdamW optimizer is used to update the model weights. It is an extension of Adam with weight decay for regularization. This optimizer adjusts the learning rate during training to ensure efficient convergence.

Learning Rate Scheduler: A ReduceLROnPlateau scheduler is applied, which reduces the learning rate when the validation loss plateaus. This helps the model fine-tune its performance as training progresses.

Gradient Clipping: To prevent exploding gradients (which can cause instability during training), gradient clipping is applied, ensuring that the gradients do not exceed a specified threshold (max_norm=5.0).

4. Hyperparameter Tuning with Optuna
The model’s architecture and training process are optimized using Optuna, a hyperparameter optimization framework.
Tunable hyperparameters include:
Number of hidden units in the LSTM layers (hidden_layer_size).
Dropout rate (dropout).
Learning rate (learning_rate).
Optuna uses TPE (Tree-structured Parzen Estimator) to explore different hyperparameter combinations and minimize the validation loss.
5. Recursive Forecasting
After training, the model is used for recursive forecasting to predict the next 30 days of Bitcoin prices.
The model takes the last sequence of data (240 days), predicts the next day’s price, and then uses this predicted price to update the input sequence, along with recalculating technical indicators dynamically (such as MACD, RSI, Bollinger Bands, etc.).
This process continues for 30 iterations to predict the prices for the next 30 days.
6. Evaluation Metrics
The model’s performance is evaluated using several common regression metrics:
Mean Squared Error (MSE): Measures the average squared difference between predicted and actual prices.
Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual prices.
Mean Absolute Percentage Error (MAPE): Measures the average percentage difference between predicted and actual prices.
R² Score: Indicates how well the predictions explain the variance in the actual prices (1.0 is perfect prediction, 0.0 means the model performs no better than the mean).
7. Visualization
The model visualizes its predictions by plotting:
Historical Bitcoin prices for the last 5 years.
Predictions for the validation period (using data not seen by the model during training).
Predictions for the next 30 days, showing expected future prices.
Model Summary:
Type: LSTM-based RNN with time series forecasting.
Input Features: Technical indicators + price data (8 features).
Output: One-day-ahead Bitcoin price prediction (and recursive predictions for the next 30 days).
Optimization: AdamW optimizer, Optuna hyperparameter tuning.
Regularization: Dropout, gradient clipping, learning rate scheduling.
Target: Predict the future price of Bitcoin using sequential historical data with advanced technical indicators.





