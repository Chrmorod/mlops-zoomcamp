import argparse
import pandas as pd
import pickle
import os

def read_dataframe(input_file,year, month):
    categorical = ['PULocationID', 'DOLocationID']
    print(f'Reading data from {input_file}...')
    df = pd.read_parquet(input_file)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df


def prepare_dictionaries(df):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    return dicts


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv,model


def save_results(df, y_pred, output_file):
    print(f'Saving results to {output_file}...')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction'] = y_pred
    #df_result['diff'] = df['duration'] - df_result['prediction']

    df_result.to_parquet(f"./predictions/{output_file}",engine='pyarrow',compression=None,index=False)
    return print(os.path.getsize(f'./predictions/{output_file}') / 1024**2, "MB")

def main(year, month):
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"predictions_yellow_{year:04d}-{month:02d}.parquet"

    df = read_dataframe(input_file,year, month)
    dicts = prepare_dictionaries(df)

    dv,model = load_model()
    print('Making predictions...')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    save_results(df, y_pred, output_file)
    return print(f"Mean predicted duration: {y_pred.mean()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ride Duration Prediction CLI")
    parser.add_argument("--year", type=int, required=True, help="Year of the dataset (e.g. 2023)")
    parser.add_argument("--month", type=int, required=True, help="Month of the dataset (e.g. 4)")

    args = parser.parse_args()
    main(args.year, args.month)
