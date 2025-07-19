if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import mlflow 

@transformer
def transform(data, *args, **kwargs):
    import numpy as np
    import pandas as pd
    import warnings
    from datetime import datetime 

    warnings.simplefilter(action='ignore', category=FutureWarning)

    data.index = pd.to_datetime(data.index)

    n_days = int(kwargs.get('n_days', 30))
    n_simulations = int(kwargs.get('n_simulations', 1000))

    prices = data['Close']
    last_price = float(prices.iloc[-1])

    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    dt = 1 / 252  # Un día de mercado
    
    def stock_monte_carlo(start_price, days, mu, sigma):
        price = np.zeros(days)
        price[0] = start_price
        for t in range(1, days):
            shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
            drift = mu * dt
            price[t] = price[t - 1] + (price[t - 1] * (drift + shock))
        return price

    simulations = np.zeros((n_days, n_simulations))
    for i in range(n_simulations):
        simulations[:, i] = stock_monte_carlo(last_price, n_days, mu, sigma)

    last_date = prices.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    simulations_df = pd.DataFrame(simulations, index=future_dates)

    last_simulated_price = simulations_df.iloc[-1]
    average_expected_price = last_simulated_price.mean()
    conf_interval = np.percentile(last_simulated_price, [5, 95])
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(f"montecarlo_{n_days}d_{n_simulations}s_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    with mlflow.start_run(run_name=f"montecarlo_{n_days}d_{n_simulations}s"):
        mlflow.log_param("n_days", n_days)
        mlflow.log_param("n_simulations", n_simulations)
        mlflow.log_param("expected_price", average_expected_price)
        mlflow.log_param("mu", mu)
        mlflow.log_param("sigma", sigma)

        mlflow.log_metric("average_expected_price", average_expected_price)
        mlflow.log_metric("conf_interval_low", conf_interval[0])
        mlflow.log_metric("conf_interval_high", conf_interval[1])

    print(f"IC 90%: ${conf_interval[0]:.2f} - ${conf_interval[1]:.2f}")

    return {
        'last_expected_price_montecarlo': average_expected_price, #same value in this case
        'average_expected_price_montecarlo': average_expected_price
    }