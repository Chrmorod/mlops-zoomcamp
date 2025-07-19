import mlflow

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, data_2, data_3, *args, **kwargs):
    expected_price_MonteCarlo = data.get('last_expected_price_montecarlo')
    expected_price_ARIMA = data_2.get('last_expected_price_arima')
    expected_price_LSTM = data_3.get('last_expected_price_lstm')
    average_expected_price_MonteCarlo = data.get('average_expected_price_montecarlo')
    average_expected_price_ARIMA = data_2.get('average_expected_price_arima')
    average_expected_price_LSTM = data_3.get('average_expected_price_lstm')

    hybrid_expected_price = (
        float(expected_price_MonteCarlo) + 
        float(expected_price_ARIMA) + 
        float(expected_price_LSTM)
    ) / 3

    hybrid_average_expected_price = (
        float(average_expected_price_MonteCarlo) + 
        float(average_expected_price_ARIMA) + 
        float(average_expected_price_LSTM)
    ) / 3
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("hybrid_prediction_price")

    with mlflow.start_run():
        mlflow.log_param("model_1", "MonteCarlo")
        mlflow.log_param("model_2", "ARIMA")
        mlflow.log_param("model_3", "LSTM")

        mlflow.log_metric("expected_price_montecarlo", expected_price_MonteCarlo)
        mlflow.log_metric("expected_price_arima", expected_price_ARIMA)
        mlflow.log_metric("expected_price_lstm", expected_price_LSTM)
        mlflow.log_metric("hybrid_expected_price", hybrid_expected_price)

        mlflow.log_metric("average_expected_price_montecarlo", average_expected_price_MonteCarlo)
        mlflow.log_metric("average_expected_price_arima", average_expected_price_ARIMA)
        mlflow.log_metric("average_expected_price_lstm", average_expected_price_LSTM)
        mlflow.log_metric("hybrid_average_expected_price", hybrid_average_expected_price)

    return {
        'hybrid_expected_price': hybrid_expected_price,
        'hybrid_average_expected_price': hybrid_average_expected_price
    }
