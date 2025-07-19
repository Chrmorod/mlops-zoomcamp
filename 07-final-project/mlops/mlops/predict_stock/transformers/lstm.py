if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import mlflow
import warnings

@transformer
def transform(data):
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense

    warnings.filterwarnings("ignore") 
    
    data.index = pd.to_datetime(data.index)
    close_prices = data['Close'].values.reshape(-1,1)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)
    
    last_days = 60
    train_len = int(len(scaled_data) * 0.8)
    
    x_train, y_train = [], []
    for i in range(last_days, train_len):
        x_train.append(scaled_data[i-last_days:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)
    
    days_to_predict = 30
    last_60_days = scaled_data[-last_days:]
    x_input = last_60_days.reshape(1, last_days, 1)
    
    predictions = []
    for _ in range(days_to_predict):
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0,0])
        x_input = np.append(x_input[:,1:,:], pred.reshape(1,1,1), axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    
    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
    
    forecast_df = pd.DataFrame(predictions, index=future_dates, columns=['Prediction'])
    
    expected_price = forecast_df['Prediction'].iloc[-1]
    average_expected_price = forecast_df['Prediction'].mean()
    
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment(f"Transformer_LSTM_{days_to_predict}d_{last_date.strftime('%Y%m%d')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        with mlflow.start_run(run_name=f"Run_{last_date.strftime('%Y-%m-%d')}"):
            mlflow.log_param("model", "LSTM (transformer alternative)")
            mlflow.log_param("days_to_predict", days_to_predict)
            mlflow.log_param("last_date", str(last_date))
            
            mlflow.log_metric("expected_price", expected_price)
            mlflow.log_metric("average_expected_price", average_expected_price)
    except Exception as e:
        print(f"MLflow error: {e}")
    
    return {
        'last_expected_price_lstm': float(expected_price),
        'average_expected_price_lstm': float(average_expected_price)
    }
