if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def extract_data(**kwargs):
    from datetime import datetime
    import yfinance as yf
    import pandas as pd
    import mlflow
    import os

    stock = kwargs.get('stock') or os.getenv('STOCK', 'NT5.F')
    year_back = int(kwargs.get('year_back') or os.getenv('YEAR_BACK', 4))
    
    #stock = kwargs['stock']
    #year_back = int(kwargs['year_back'])
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(f"extract_data__{stock}_finance_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    end = datetime.now()
    start = datetime(end.year - year_back, end.month, end.day)

    df = yf.download(stock, start=start, end=end)

    df_close = df[['Close']].copy().reset_index()

    with mlflow.start_run(run_name=f"extract_{stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("stock", stock)
        mlflow.log_param("year_back", year_back)
        mlflow.log_param("start_date", start.strftime('%Y-%m-%d'))
        mlflow.log_param("end_date", end.strftime('%Y-%m-%d'))
        mlflow.log_metric("num_rows", df_close.shape[0])
        mlflow.log_metric("num_columns", df_close.shape[1])

    if isinstance(df_close.columns, pd.MultiIndex):
        df_close.columns = df_close.columns.get_level_values(0)  #first level

    df_close.columns = [str(c) for c in df_close.columns]

    return df_close